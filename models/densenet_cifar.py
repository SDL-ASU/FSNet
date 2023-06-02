import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math
import pdb
from .dwsconv import Conv2d_dws, conv3x3, conv3x3_dws, conv1x1_dws


class Bottleneck_cifar(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck_cifar, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class Bottleneck_cifar_dws(nn.Module):
    def __init__(self, nChannels, growthRate, compression_ratio, binary_dws):
        super(Bottleneck_cifar_dws, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = Conv2d_dws(nChannels, interChannels, compression_ratio, kernel_size=1,
                               binary_dws=binary_dws, bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = Conv2d_dws(interChannels, growthRate, compression_ratio, kernel_size=3,
                               binary_dws=binary_dws, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer_cifar(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer_cifar, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer_cifar_dws(nn.Module):
    def __init__(self, nChannels, growthRate, compression_ratio, binary_dws):
        super(SingleLayer_cifar_dws, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = Conv2d_dws(nChannels, growthRate, compression_ratio, kernel_size=3,
                               binary_dws=binary_dws, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition_cifar(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition_cifar, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class Transition_cifar_dws(nn.Module):
    def __init__(self, nChannels, nOutChannels, compression_ratio, binary_dws):
        super(Transition_cifar_dws, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = Conv2d_dws(nChannels, nOutChannels, compression_ratio, kernel_size=1,
                               binary_dws=binary_dws, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet_cifar(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet_cifar, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition_cifar(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition_cifar(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck_cifar(nChannels, growthRate))
            else:
                layers.append(SingleLayer_cifar(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out

class DenseNet_cifar_dws(nn.Module):
    def __init__(self,compression_ratio,binary_dws,growthRate,depth,reduction,nClasses,bottleneck):
        super(DenseNet_cifar_dws, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = Conv2d_dws(3, nChannels, compression_ratio, kernel_size=3, binary_dws=binary_dws, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, compression_ratio, binary_dws)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition_cifar_dws(nChannels, nOutChannels, compression_ratio, binary_dws)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, compression_ratio, binary_dws)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition_cifar_dws(nChannels, nOutChannels, compression_ratio, binary_dws)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, compression_ratio, binary_dws)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, Conv2d_dws):
                n = m.reduced_weights #m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.filtermap.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, compression_ratio, binary_dws):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck_cifar_dws(nChannels, growthRate, compression_ratio, binary_dws))
            else:
                layers.append(SingleLayer_cifar_dws(nChannels, growthRate, compression_ratio, binary_dws))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out

def densenet100_gr12_cifar(reduction=1):
    return DenseNet_cifar(growthRate=12, depth=100, reduction=reduction,bottleneck=True, nClasses=10)

def densenet_cifar_fm100_gr12(compression_ratio, binary_dws, reduction=1):
    #pdb.set_trace()
    return DenseNet_cifar_dws(compression_ratio, binary_dws, growthRate=12, depth=100, \
                        reduction=reduction,bottleneck=True, nClasses=10)

def densenet100_gr24_cifar(reduction=1):
    return DenseNet_cifar(growthRate=24, depth=100, reduction=reduction,bottleneck=True, nClasses=10)

def densenet_cifar_fm100_gr24(compression_ratio, binary_dws, reduction=1):
    #pdb.set_trace()
    return DenseNet_cifar_dws(compression_ratio, binary_dws, growthRate=24, depth=100, \
                        reduction=reduction,bottleneck=True, nClasses=10)

