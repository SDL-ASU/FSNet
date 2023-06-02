'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.distributed as dist

import os
import argparse

from models.densenet import Conv2d_filtermap as Conv2d_filtermap_densenet
from models.filtermap_net import Conv2d_filtermap as Conv2d_filtermap_resnet_cifar
from models.resnet_imagenet import Conv2d_filtermap as Conv2d_filtermap_resnet_imagenet
from models import *
from utils import progress_bar, print_flush_write
from net_utils import construct_network, get_dataloader
import re
import pdb

#pdb.set_trace()
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--has_logfile', action='store_true', help='log file')
parser.add_argument('--dataset', '-dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--imagenet', '-imagenet', type=str, default='imagenet_path_to_be_set', help='path to imagenet dataset')

parser.add_argument('--lr', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batchsize', '-b', default=128, type=int, metavar='N',
                    help='mini-batch size (1 = pure stochastic) default values for cifar, svhn, imagenet are 128, 200, 256')
parser.add_argument('--compression1x1', '-compression1x1', default=1, type=int, \
                    help='compression ratio for 1x1 convolution layers')
parser.add_argument('--binary_filtermap', '-bfm', action='store_true', help='using binary filtermap')

#resume
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--load_from_folder', '-l', type=str, default='', help='location of the checkpoint to load model')
parser.add_argument('--resume_last_epoch', action='store_true', help='resume from last epoch')

parser.add_argument('--lr_short_epoch', '-lr_short_epoch', action='store_true', help='using lr policy [82,123]')
parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))

parser.add_argument('--checkpoint_folder_fm', type=str, default='', help='location of the checkpoint of the fm model')
parser.add_argument('--checkpoint_folder_ori', type=str, default='', help='location of the checkpoint of the original model') 

parser.add_argument('--transform_net', '-t', action='store_true', help='using transform net')

#arch
parser.add_argument('--fm_arch', '-fm_arch', type=str, default='resnet18', help='network architecture of fm')
parser.add_argument('--ori_arch', '-ori_arch', type=str, default='resnet_fm18', help='network architecture of ori')
parser.add_argument("--cycle", '-cycle', type=int, default=90, help='number of cycles for training')
parser.add_argument("--epoches", '-e', type=int, default=90, help='number of epoches for training')
parser.add_argument("--epoches_stage", '-epoches_stage', type=int, default=30, \
                    help='number of epoches for each stage (before and after decreasing the lr) of training on imagenet')
parser.add_argument("--workers", '-w', type=int, default=4, help='number of workers for training')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--evaluate', '-evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--progress_bar', action='store_true', help='turn on progress bar')
parser.add_argument('--opt_compressed_weight', action='store_true', help='optimize the compressed weight')
parser.add_argument('--opt_transformation', action='store_true', help='optimize the transformation')

parser.add_argument('--init_filtermap', '-init_fm', action='store_true', help='initialize the filtermap_net')
parser.add_argument('--logfile_name', '-logfile_name', type=str, default='', help='name of logfile')

args = parser.parse_args()

args.distributed = args.world_size > 1
if args.distributed:
   dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#log
if args.logfile_name!='':
   logfile_name = args.logfile_name
else:
   logfile_name = 'quantization.log'
logfile = open(logfile_name,"a+")

# Data
#print_flush_write(args.has_logfile, logfile, '==> Preparing data..\n')

trainloader, testloader = get_dataloader(args, logfile)
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def valid_net_parameters(net_params):
    for param_iter in self.named_parameters():
        yield param_iter

def net_parameters_size(net):
    #print('Total model parameters: {}'.format(np.sum([np.prod(list(w.size())) for w in model.parameters()]))
    #sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_count = 0
    for param in net.parameters():
        #if not isinstance(param.data,torch.cuda.LongTensor): filtermap ids are not registered as parameter
        if param.data.type() != 'torch.ByteTensor':
           parameter_count  = parameter_count + torch.numel(param.data)
        else:
           parameter_count  = parameter_count + torch.numel(param.data)/4.0
    return parameter_count

def net_parameters_size_state_dict(net):
    parameter_count = 0
    for name, param in net.state_dict().items():
        parameter_count  = parameter_count + torch.numel(param.data)
    return parameter_count


def get_conv_layers(m, Conv2d_filtermap_module):
    layers = []
    if isinstance(m, Conv2d_filtermap_module):
        layers.append(m)
    for child in m.children():
        layers += get_conv_layers(child, Conv2d_filtermap_module)
    return layers

def quantization(net,args,symmetric=False,quantize_fm=False):
    if quantize_fm:
       arch = args.fm_arch
    else: 
       arch = args.ori_arch
    dataset = args.dataset
    if 'fm' in arch:
       compression1x1 = args.compression1x1
       binary_filtermap = args.binary_filtermap
       quant_net = eval(arch + '(compression1x1, binary_filtermap)')
    else:
       quant_net = eval(arch + '()')
    #quant_net.load_state_dict(net.state_dict())
    if quantize_fm:
       if 'densenet' in arch:
          Conv2d_filtermap_module = Conv2d_filtermap_densenet
       elif 'resnet' in arch:
            if dataset == 'imagenet':
               Conv2d_filtermap_module = Conv2d_filtermap_resnet_imagenet
            elif dataset == 'cifar10':
               Conv2d_filtermap_module = Conv2d_filtermap_resnet_cifar
            else:
               Conv2d_filtermap_module = Conv2d_filtermap_resnet_cifar
       else:
            Conv2d_filtermap_module = Conv2d_filtermap_resnet_cifar

       convs = get_conv_layers(net, Conv2d_filtermap_module)
       quant_convs = get_conv_layers(quant_net, Conv2d_filtermap_module)
       print('%d conv layers will be quantized\n' % (len(convs)))

    ori_dict = net.state_dict()
    quant_dict = quant_net.state_dict()
    quant_net_size = 0

    #pdb.set_trace()
    #begin the quantization now
    quant_level = 256 #255
    for name, param in ori_dict.items():
        param = param.to(device)
        if quantize_fm:
           if 'filtermap' in name:
              fm = param
              fm_max = torch.max(fm) 
              fm_min = torch.min(fm)
              if not symmetric:
                 ss = torch.div(fm_max-fm_min,quant_level-1)
                 m0 = torch.round(torch.floor(torch.div(fm_min,ss)))
                 fm_min = torch.mul(ss,m0)
                 #ss = torch.div(fm_max-fm_min,255)
              else:
                 fm_max = torch.abs(fm_max)
                 fm_min = torch.abs(fm_min)
                 if fm_max >= fm_min:
                    fm_min = -fm_max
                    ss = torch.div(fm_max-fm_min,quant_level-1)
                 else:
                    fm_max = fm_min
                    fm_min = -fm_min
                    ss = torch.div(fm_max-fm_min,quant_level-1)
              fm_diff = fm - fm_min
              fm_diff_ind = torch.round(torch.div(fm_diff,ss))
              quant_dict[name].copy_(fm_min + fm_diff_ind*ss)
              quant_net_size = quant_net_size + (torch.numel(param)/4.0+2)
              #print(name+' is quantized\n')
           else:
              quant_dict[name].copy_(param)
              quant_net_size = quant_net_size + torch.numel(param)
        else:
            if 'conv' in name or 'fc' in name:
               fm = param
               fm_max = torch.max(fm) 
               fm_min = torch.min(fm)
               if not (fm_max >= 0 and fm_min <= 0):
                  print_flush_write(args.has_logfile, logfile, 'fm_max >= 0 and fm_min <= 0 does not hold')
               #print_flush_write(args.has_logfile, logfile, name+ \
               #                  ':(max,min) is (%.3f,%.3f) \n' %(fm_max,fm_min))
               
               #if name == 'layer1.0.conv1.weight' or name == 'layer2.1.conv1.weight' or name == 'fc.weight':
               #   import numpy
               #   import scipy.io
               #   scipy.io.savemat(name+'.mat',dict(x=fm.cpu().numpy()))
               #pdb.set_trace()
               
               if not symmetric:
                  #tf.quantize simulation
                  #######################################################################
                  ss =  torch.div((fm_max-fm_min)*quant_level/(quant_level-1),quant_level) 
                  fm_diff = fm - fm_min
                  fm_diff_ind = torch.round(torch.div(fm_diff,ss))
                  m0 = torch.round(torch.div(fm_min,ss))
                  ind_m0 = torch.round(torch.div(fm,ss))-m0-128 #fm_diff_ind - 128
                  ind_m0 = torch.max(ind_m0.to(device),torch.Tensor([-128]).to(device)) 
                  ind_m0 = torch.min(ind_m0.to(device),torch.Tensor([127]).to(device)) 
                  
                  quant_dict[name].copy_(fm_min + (ind_m0+128)*ss)

                  #######################################################################

                  #ss = torch.div(fm_max-fm_min,quant_level-1)
                  #m0 = torch.round(torch.div(fm_min,ss)) #torch.round(torch.floor(torch.div(fm_min,ss)))
                  #fm_min = torch.mul(ss,m0)
                  #fm_diff = fm - fm_min
                  #fm_diff_ind = torch.round(torch.div(fm_diff,ss))
                  #fm_diff_ind = torch.max(fm_diff_ind.to(device),torch.Tensor([0]).to(device))
                  #fm_diff_ind = torch.min(fm_diff_ind.to(device),torch.Tensor([255]).to(device))

                  #fm_diff_ind = fm_diff_ind-128
                  #fm_diff_ind = fm_diff_ind.to(torch.int8)

                  #m0 = m0+128
                  #if m0 < -128 or m0 > 127:
                  #   print('m0 = %d out of range\n'%(m0))
                  #if m0 < -128:
                  #   m0 = -128
                  #if m0 > 127:
                  #   m0 = 127
                  #m0 = m0.to(torch.int8)
                  #pdb.set_trace()
                  #fm_diff_ind = fm_diff_ind.to(torch.int16)
                  #m0 = m0.to(torch.int16)
                  #ind_m0 = fm_diff_ind+m0
                  #ind_m0 = ind_m0.to(torch.float).to(device)

                  #quant_dict[name].copy_(ind_m0*ss)
               else:
                  fm_max = torch.abs(fm_max)
                  fm_min = torch.abs(fm_min)
                  if fm_max >= fm_min:
                     fm_min = -fm_max
                     ss = torch.div(fm_max-fm_min,255)
                  else:
                     fm_max = fm_min
                     fm_min = -fm_min
                     ss = torch.div(fm_max-fm_min,255)
                     fm_diff = fm - fm_min
                     fm_diff_ind = torch.round(torch.div(fm_diff,ss))
                     quant_dict[name].copy_(fm_min + fm_diff_ind*ss)
               quant_net_size = quant_net_size + (torch.numel(param)/4.0+2)
               #print(name+' is quantized\n')
            else:
               quant_dict[name].copy_(param)
               quant_net_size = quant_net_size + torch.numel(param)
    return quant_net, quant_net_size

# Model
if os.path.isdir(args.checkpoint_folder_ori) or os.path.isdir(args.checkpoint_folder_fm):
   #ori_net = eval(args.ori_arch + '()')
   ori_checkpoint = torch.load(args.checkpoint_folder_ori+'/ckpt.t7')
   ori_net = ori_checkpoint['net']
   ori_net_size = net_parameters_size_state_dict(ori_net)
   print('ori_net_size is %d.\n' %(ori_net_size) ) 
   #fm_checkpoint = torch.load(args.checkpoint_folder_fm+'/ckpt.t7')
   #fm_net = fm_checkpoint['net']
   #fm_net_size = net_parameters_size_state_dict(fm_net)
   #print('fm_net_size is %d. Begin quantization \n' %(fm_net_size) ) 
   #pdb.set_trace()
   #quant_fm_net, quant_fm_net_size = quantization(fm_net, args,symmetric=False,quantize_fm=True)
   quant_net, quant_net_size = quantization(ori_net, args, symmetric=False)
   print('quantization finished, quant_net_size is %d. Begin testing\n' %(quant_net_size) )
   #print_flush_write(args.has_logfile, logfile, 'loading from last saved model, start_epoch : %d, start_cycle : %d\n' \
   #            %(start_epoch, start_cycle))
#pdb.set_trace()
#del fm_net
#torch.cuda.empty_cache()
if 'quant_fm_net' in globals():
   quant_net = quant_fm_net
if 'quant_fm_net_size' in globals():
   quant_net_size = quant_fm_net_size

criterion = nn.CrossEntropyLoss()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(epoch, net, *save_args):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    test_prec1 = 0
    test_prec5 = 0
    #if epoch < 3:
    #   return

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        #pdb.set_trace()
        _outputs = net(inputs)
        #outputs_ref_resnet = ref_resnet(inputs)
        if isinstance(_outputs, list):
           outputs = _outputs[0]
        else:
           outputs = _outputs

        loss = criterion(outputs, targets)

        test_loss += loss.item()

        if args.dataset != "imagenet":
           _, predicted = torch.max(outputs.data, 1)
           total += targets.size(0)
           correct += predicted.eq(targets).sum().item()
        else:
           prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
           test_prec1 += prec1[0]
           test_prec5 += prec5[0]

        if args.progress_bar:
           if args.dataset != "imagenet":
              progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
           else:
              progress_bar(batch_idx, len(testloader), 'Loss: %.3f | top-1: %.3f%% | top-5: %.3f%% '
                   % (test_loss/(batch_idx+1), test_prec1/(batch_idx+1), test_prec5/(batch_idx+1)))
        num_batches = num_batches + 1

    if args.dataset != "imagenet":
       print_flush_write(args.has_logfile, logfile, \
               'test_loss: %.3f, test_acc: %.3f%% (%d/%d)\n' % (test_loss/num_batches, 100.*correct/total, correct, total))
    else:
       print_flush_write(args.has_logfile, logfile, \
               'test_loss: %.3f, top-1: %.3f%%, top-5: %.3f%%\n' % (test_loss/num_batches, test_prec1/num_batches, \
             test_prec5/num_batches))
    # Save checkpoint.
    if args.dataset != "imagenet":
       acc = 100.*correct/total
    else:
       top1 = test_prec1/num_batches
       top5 = test_prec5/num_batches

if args.evaluate:
   #print('testing the ori_net:\n') 
   #ori_net = ori_net.to(device)
   #if device == 'cuda':
   #   if args.distributed:
   #      ori_net = torch.nn.parallel.DistributedDataParallel(ori_net)
   #   else:
   #      ori_net = torch.nn.DataParallel(ori_net, device_ids=range(torch.cuda.device_count()))
   #   cudnn.benchmark = True
   #test(0,ori_net)
   
   #print('testing the fm_net:\n') 
   #fm_net = fm_net.to(device)
   #if device == 'cuda':
   #   if args.distributed:
   #      fm_net = torch.nn.parallel.DistributedDataParallel(fm_net)
   #   else:
   #      fm_net = torch.nn.DataParallel(fm_net, device_ids=range(torch.cuda.device_count()))
   #   cudnn.benchmark = True
   #test(0,fm_net)
   
   #del fm_net
   print('testing the quant_net:\n')
   quant_net = quant_net.to(device)
   if device == 'cuda':
      if args.distributed:
         quant_net = torch.nn.parallel.DistributedDataParallel(quant_net)
      else:
         quant_net = torch.nn.DataParallel(quant_net, device_ids=range(torch.cuda.device_count()))
      cudnn.benchmark = True
   test(0,quant_net)
   quit()
