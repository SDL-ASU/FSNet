'''Filtermap Net in PyTorch.

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
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

class _ConvNd_dws(torch.nn.Module):

    def __init__(self, in_channels, out_channels, compression_ratio, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd_dws, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups



        #key idea: sample 1D segments from filtermap, and then reshape each sampled 1D as a 3D filter for convolution
        #total_weights: total number of weights in the conv before compression

        total_weights = out_channels*kernel_size[0]*kernel_size[1]*in_channels

        self.reduced_weights = int(total_weights/compression_ratio)

        #filtermap is the condensed representation for 3D filters
        self.filtermap = Parameter(torch.Tensor(self.reduced_weights))

        #filter_grid specifies the location of each filter to be sampled from filtermap
        #a floating number is used to specify the location of each sampled filter in filtermap,
        #so the size of filter_grid is of out_channels

        #self.filter_grid = Parameter(torch.Tensor(out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        #self.weight.data.uniform_(-stdv, stdv)

        #pdb.set_trace()
        self.filtermap.data.uniform_(-stdv, stdv)
        #step = 1.0/(self.out_channels-1)
        #for k in range(0,self.out_channels):
        #    seg_end = torch.tensor(k*step)
        #    if k == 0:
        #       seg_end = torch.tensor(1e-6)
        #    if k == self.out_channels-1:
        #       seg_end = torch.tensor(1-(1e-6))
        #
        #    self.filter_grid.data[k] = -1*torch.log((1.0/seg_end)-1)
        ## self.filter_grid.data.fill_()
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d_dws(_ConvNd_dws):
    def __init__(self, in_channels, out_channels, compression_ratio, kernel_size, binary_dws = False, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(Conv2d_dws, self).__init__(
            in_channels, out_channels, compression_ratio, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        self.binary_dws = binary_dws
        self.reset_parameters1()
        #pdb.set_trace()
        #self.conv_weight = Parameter(torch.Tensor(out_channels, self.in_channels // self.groups, \
        #              self.kernel_size[0], self.kernel_size[1]))
        #self.conv_weight.requires_grad = False
        #self.conv_weight.cuda()
    def reset_parameters1(self):
        step = self.reduced_weights/(self.out_channels-1)
        filter_size = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        ids0 = torch.Tensor(range(0,filter_size))
        ids = ids0

        for c in range(1,self.out_channels):
            seg_end = torch.tensor(c*step)
            ids_c = ids0 + seg_end
            ids = torch.cat((ids,ids_c),0)
        #pdb.set_trace()
        #rp = torch.randperm(ids.size()[0])
        #ids = ids[rp]
        ids = ids.long()
        self.ids = ids
    def extract_filters(self):
        filtermap_pad = torch.cat([self.filtermap, self.filtermap],0)
        ids = self.ids.cuda()
        conv_weight = filtermap_pad.view(-1,1).index_select(0,ids)
        conv_weight = conv_weight.view(self.out_channels,self.in_channels,*self.kernel_size)
        return conv_weight
    #def extract_filters_dws(self):
    #    # pdb.set_trace()
    #    filter_size = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
    #    filtermap_pad = torch.cat([self.filtermap, self.filtermap],0)
    #    sampled_filter = []
    #    pdb.set_trace()
    #    #if torch.isnan(self.filter_grid).any():
    #    #   print('nan elements in filter_grid\n')   
    #    for i in range(self.out_channels):
    #        grid = torch.sigmoid(self.filter_grid[i]) * self.reduced_weights
    #        low_grid = torch.round(grid)
    #        high_grid = low_grid + 1
    #        sampled_filter.append(
    #            (grid - low_grid) * filtermap_pad[low_grid.int(): low_grid.int() + filter_size]
    #            + (high_grid - grid) * filtermap_pad[high_grid.int(): high_grid.int() + filter_size])
    #    sampled_filter = torch.stack(sampled_filter)
    #    return sampled_filter.reshape(self.out_channels, self.in_channels, *self.kernel_size)


    def forward(self, input):
        conv_weight = self.extract_filters()
        out = F.conv2d(input, conv_weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_dws(in_planes, out_planes, compression_ratio, binary_dws = False, stride=1):
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return Conv2d_dws(in_planes, out_planes, compression_ratio, kernel_size=3, \
                            binary_dws=binary_dws, stride=stride, padding=1, bias=False)

def conv1x1_dws(in_planes, out_planes, compression_ratio, binary_dws = False, stride=1):
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return Conv2d_dws(in_planes, out_planes, compression_ratio, kernel_size=1, \
                            binary_dws=binary_dws, stride=stride, padding=1, bias=False)

