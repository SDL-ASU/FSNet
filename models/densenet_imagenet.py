'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

# coding=utf-8
import math
#import torch
from torch.nn.parameter import Parameter
#from .. import functional as F
#from torch.module import Module
#from torch.nn.module.utils import _single, _pair, _triple
from numpy import prod
import collections
from itertools import repeat
import pdb
from .dwsconv import Conv2d_dws, conv3x3, conv3x3_dws, conv1x1_dws

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out

class Bottleneck_dws(nn.Module):
    def __init__(self, in_planes, growth_rate, compression_ratio, binary_dws):
        super(Bottleneck_dws, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = Conv2d_dws(in_planes, 4*growth_rate, compression_ratio, kernel_size=1, \
                binary_dws=binary_dws, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = Conv2d_dws(4*growth_rate, growth_rate, compression_ratio, \
                kernel_size=3, binary_dws=binary_dws, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class Transition_dws(nn.Module):
    def __init__(self, in_planes, out_planes, compression_ratio, binary_dws):
        super(Transition_dws, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = Conv2d_dws(in_planes, out_planes, compression_ratio, kernel_size=1, \
                binary_dws=binary_dws, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet_imagenet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet_imagenet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class DenseNet_imagenet_dws(nn.Module):
    def __init__(self, block, nblocks, compression_ratio, binary_dws, growth_rate=12, \
            reduction=0.5, num_classes=10):
        super(DenseNet_imagenet_dws, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = Conv2d_dws(3, num_planes, compression_ratio, \
                kernel_size=3, binary_dws=binary_dws, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], \
                compression_ratio, binary_dws)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition_dws(num_planes, out_planes, compression_ratio, binary_dws)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], \
                compression_ratio, binary_dws)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition_dws(num_planes, out_planes, compression_ratio, binary_dws)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], \
                compression_ratio, binary_dws)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition_dws(num_planes, out_planes, compression_ratio, binary_dws)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], \
                compression_ratio, binary_dws)
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock, compression_ratio, binary_dws):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, compression_ratio, binary_dws))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def densenet121_imagenet():
    return DenseNet_imagenet(Bottleneck, [6,12,24,16], growth_rate=32)

def densenet169_imagenet():
    return DenseNet_imagenet(Bottleneck, [6,12,32,32], growth_rate=32)

def densenet201_imagenet():
    return DenseNet_imagenet(Bottleneck, [6,12,48,32], growth_rate=32)

def densenet161_imagenet():
    return DenseNet_imagenet(Bottleneck, [6,12,36,24], growth_rate=48)

#def densenet_cifar():
#    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def densenet_fm121_imagenet(compression_ratio, binary_dws):
    return DenseNet_imagenet_dws(Bottleneck_dws, [6,12,24,16], compression_ratio, \
            binary_dws, growth_rate=32)

def densenet_fm169_imagenet(compression_ratio, binary_dws):
    return DenseNet_imagenet_dws(Bottleneck_dws, [6,12,32,32], compression_ratio, \
            binary_dws, growth_rate=32)

def densenet_fm201_imagenet(compression_ratio, binary_dws):
    return DenseNet_imagenet_dws(Bottleneck_dws, [6,12,48,32], compression_ratio, \
            binary_dws, growth_rate=32)

def densenet_fm161_imagenet(compression_ratio, binary_dws):
    return DenseNet_imagenet_dws(Bottleneck_dws, [6,12,36,24], compression_ratio, \
            binary_dws, growth_rate=48)

#def densenet_fm_cifar(compression_ratio, binary_dws):
#    return DenseNet_dws(Bottleneck_dws, [6,12,24,16], compression_ratio, \
#            binary_dws, growth_rate=12)

def test_densenet():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y)

# test_densenet()
