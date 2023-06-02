'''ResNet in PyTorch.

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

#from transform_conv import _Transform_ConvNd, Transform_Conv2d
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

class _Transform_ConvNd(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, compression_rate):
        super(_Transform_ConvNd, self).__init__()
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
        
        self.compressed_channels = out_channels // compression_rate
        
        self.transform_mat = Parameter(torch.Tensor(self.compressed_channels, out_channels // groups))
        self.compressed_weight = Parameter(torch.Tensor(in_channels * kernel_size[0] * kernel_size[1], \
               self.compressed_channels))       
        
        #self.input_weight = Parameter(torch.Tensor(in_channels // groups * kernel_size[0] * kernel_size[1], out_channels))
        #self.transform_mat = Parameter(torch.Tensor(out_channels, self.compressed_channels))
        #self.transform_back_mat = Parameter(torch.Tensor(self.compressed_channels, out_channels))
        #if transposed:
            #self.weight = Parameter(torch.Tensor(
            #    in_channels, out_channels // groups, *kernel_size))
        #else:
            #self.weight = Parameter(torch.Tensor(
            #    out_channels, in_channels // groups, *kernel_size))
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
        
        self.compressed_weight.data.uniform_(-stdv, stdv)
        self.transform_mat.data.uniform_(-stdv, stdv)
        
        #self.input_weight.data.uniform_(-stdv, stdv)
        #self.transform_mat.data.uniform_(-stdv, stdv)
        #self.transform_back_mat.data.uniform_(-stdv, stdv) 
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

class Transform_Conv2d(_Transform_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    .. math::

        \begin{array}{ll}
        out(N_i, C_{out_j})  = bias(C_{out_j})
                       + \sum_{{k}=0}^{C_{in}-1} weight(C_{out_j}, k)  \star input(N_i, k)
        \end{array}

    where :math:`\star` is the valid 2D `cross-correlation`_ operator

    | :attr:`stride` controls the stride for the cross-correlation.
    | If :attr:`padding` is non-zero, then the input is implicitly zero-padded
      on both sides for :attr:`padding` number of points.
    | :attr:`groups` controls the connections between inputs and outputs.
      `in_channels` and `out_channels` must both be divisible by `groups`.
    |       At groups=1, all inputs are convolved to all outputs.
    |       At groups=2, the operation becomes equivalent to having two conv
                 layers side by side, each seeing half the input channels,
                 and producing half the output channels, and both subsequently
                 concatenated.
            At groups=`in_channels`, each input channel is convolved with its
                 own set of filters (of size `out_channels // in_channels`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, ref_conv_weight, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, compression_rate = 4):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        super(Transform_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, compression_rate)
        self.ref_conv_weight = ref_conv_weight
    def forward(self, input):
        #return F.conv2d(input, self.weight, self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        
        #conv_weight = torch.mm(self.compressed_weight, torch.tanh(self.transform_mat))
        conv_weight = torch.mm(self.compressed_weight, (self.transform_mat))
        
        #compressed_weight = torch.mm(self.input_weight, torch.tanh(self.transform_mat))
        #conv_weight = torch.mm(compressed_weight, torch.tanh(self.transform_back_mat))
       
        conv_weight = conv_weight.view(self.in_channels // self.groups, self.kernel_size[0], \
                self.kernel_size[1], self.out_channels);
        conv_weight = conv_weight.permute(3, 0, 1, 2)
        conv_weight = conv_weight.contiguous()
        
        fit_loss = torch.norm(conv_weight-self.ref_conv_weight,2)

        out = []
        out.append(F.conv2d(input, conv_weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups))
   
        out.append(fit_loss)
        return out        
        #return F.conv2d(input, conv_weight, self.bias, self.stride,
        #                 self.padding, self.dilation, self.groups)




def transform_conv3x3(in_planes, out_planes, ref_conv_weight, stride=1):
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return Transform_Conv2d(in_planes, out_planes, kernel_size=3, ref_conv_weight=ref_conv_weight, stride=stride, padding=1, bias=False, compression_rate = 4)


class BasicBlock_transform(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, ref_block, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = transform_conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = transform_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Transform_Conv2d(in_planes, self.expansion*planes, kernel_size=1, \
                  ref_conv_weight=ref_block.shortcut[0].weight, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PreActBlock_transform(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, ref_block, stride=1):
        super(PreActBlock_transform, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = transform_conv3x3(in_planes, planes, ref_block.conv1.weight, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = transform_conv3x3(planes, planes, ref_block.conv2.weight)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Transform_Conv2d(in_planes, self.expansion*planes, kernel_size=1, \
                      ref_conv_weight=ref_block.shortcut[0].weight, stride=stride, bias=False)
            )

    def forward(self, _input):
        if isinstance(_input, list):
           assert len(_input) == 2
           x = _input[0]
        else:
           x = _input

        out = F.relu(self.bn1(x))

        #shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        if hasattr(self, 'shortcut'):
           shortcut, shortcut_fit_loss = self.shortcut(out)
        else:
           shortcut = x
        out, conv1_fit_loss = self.conv1(out)
        out, conv2_fit_loss  = self.conv2(F.relu(self.bn2(out)))
        out += shortcut

        fit_loss = []
        if hasattr(self, 'shortcut'):
           fit_loss.extend([shortcut_fit_loss, conv1_fit_loss, conv2_fit_loss])
        else:
           fit_loss.extend([conv1_fit_loss, conv2_fit_loss])
        if isinstance(_input, list):
           _input[1].extend(fit_loss)
           return [out, _input[1]]
        else:
           return [out, fit_loss]


class Transform_ResNet(nn.Module):
    def __init__(self, block, num_blocks, ref_resnet, num_classes=10):
        super(Transform_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = transform_conv3x3(3,64,ref_resnet.conv1.weight)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0],  stride=1, order = 1, ref_resnet = ref_resnet)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, order = 2, ref_resnet = ref_resnet)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, order = 3, ref_resnet = ref_resnet)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, order = 4, ref_resnet = ref_resnet)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, order, ref_resnet):
        #get reference layer in the reference resnet (nn.Sequential)
        ref_resnet_layer = getattr(ref_resnet, 'layer'+str(order))
        block_count = 0;

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, ref_resnet_layer[block_count], stride))
            self.in_planes = planes * block.expansion
            #increment block_count
            block_count = block_count + 1
        return nn.Sequential(*layers)

    def forward(self, x):
        #pdb.set_trace()
        fit_loss = []
        out, fit_loss_conv1 = self.conv1(x)
        fit_loss.append(fit_loss_conv1)

        out = F.relu(self.bn1(out))
        out = [out, fit_loss]

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out, fit_loss = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return [out, sum(fit_loss)/len(fit_loss)]


def Transform_ResNet18(ref_resnet):
    return Transform_ResNet(BasicBlock_transform, [2,2,2,2], ref_resnet)


def set_transform_grad(net, flag):
    net.conv1.transform_mat.requires_grad = flag
    
    net.layer1[0].conv1.transform_mat.requires_grad = flag
    net.layer1[0].conv2.transform_mat.requires_grad = flag 
    net.layer1[1].conv1.transform_mat.requires_grad = flag
    net.layer1[1].conv2.transform_mat.requires_grad = flag
    
    net.layer2[0].conv1.transform_mat.requires_grad = flag
    net.layer2[0].conv2.transform_mat.requires_grad = flag
    net.layer2[0].shortcut[0].transform_mat.requires_grad = flag
    net.layer2[1].conv1.transform_mat.requires_grad = flag
    net.layer2[1].conv2.transform_mat.requires_grad = flag
    
    net.layer3[0].conv1.transform_mat.requires_grad = flag
    net.layer3[0].conv2.transform_mat.requires_grad = flag
    net.layer3[0].shortcut[0].transform_mat.requires_grad = flag
    net.layer3[1].conv1.transform_mat.requires_grad = flag
    net.layer3[1].conv2.transform_mat.requires_grad = flag

    net.layer4[0].conv1.transform_mat.requires_grad = flag
    net.layer4[0].conv2.transform_mat.requires_grad = flag
    net.layer4[0].shortcut[0].transform_mat.requires_grad = flag
    net.layer4[1].conv1.transform_mat.requires_grad = flag
    net.layer4[1].conv2.transform_mat.requires_grad = flag


def set_compressed_weight_grad(net, flag):
    net.conv1.compressed_weight.requires_grad = flag
    
    net.layer1[0].conv1.compressed_weight.requires_grad = flag
    net.layer1[0].conv2.compressed_weight.requires_grad = flag 
    net.layer1[1].conv1.compressed_weight.requires_grad = flag
    net.layer1[1].conv2.compressed_weight.requires_grad = flag
    
    net.layer2[0].conv1.compressed_weight.requires_grad = flag
    net.layer2[0].conv2.compressed_weight.requires_grad = flag
    net.layer2[0].shortcut[0].compressed_weight.requires_grad = flag
    net.layer2[1].conv1.compressed_weight.requires_grad = flag
    net.layer2[1].conv2.compressed_weight.requires_grad = flag
    
    net.layer3[0].conv1.compressed_weight.requires_grad = flag
    net.layer3[0].conv2.compressed_weight.requires_grad = flag
    net.layer3[0].shortcut[0].compressed_weight.requires_grad = flag
    net.layer3[1].conv1.compressed_weight.requires_grad = flag
    net.layer3[1].conv2.compressed_weight.requires_grad = flag

    net.layer4[0].conv1.compressed_weight.requires_grad = flag
    net.layer4[0].conv2.compressed_weight.requires_grad = flag
    net.layer4[0].shortcut[0].compressed_weight.requires_grad = flag
    net.layer4[1].conv1.compressed_weight.requires_grad = flag
    net.layer4[1].conv2.compressed_weight.requires_grad = flag


def test():
    net = Transform_ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    #pdb.set_trace()
    print(y.size())

#test()
