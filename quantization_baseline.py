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

from models.dwsconv import Conv2d_dws
from models import *
from utils import progress_bar, print_flush_write
from net_utils import construct_network, get_dataloader
from flops_counter.flops_counter import get_model_complexity_info
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
parser.add_argument('--compression_ratio', '-compression_ratio', default=1, type=float, \
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


def get_conv_layers(m, Conv2d_dws):
    layers = []
    if isinstance(m, Conv2d_dws):
        layers.append(m)
    for child in m.children():
        layers += get_conv_layers(child, Conv2d_dws)
    return layers

def quantization(net,args):
    arch = args.fm_arch
    dataset = args.dataset
    if 'fm' in arch:
       compression_ratio = args.compression_ratio
       binary_filtermap = args.binary_filtermap
       quant_net = eval(arch + '(compression_ratio, binary_filtermap)')
    else:
       quant_net = eval(arch + '()')
    #quant_net.load_state_dict(net.state_dict())
    convs = get_conv_layers(net, Conv2d_dws)
    quant_convs = get_conv_layers(quant_net, Conv2d_dws)
    print('%d conv layers will be quantized\n' % (len(convs)))

    ori_dict = net.state_dict()
    quant_dict = quant_net.state_dict()
    quant_net_size = 0
    quant_level = 256
    quant_ratio = 4.0 #32bit/8bit = 4
    for name, param in ori_dict.items():
        if 'conv' in name or 'fc' in name:
        #if 'filtermap' in name or 'fc' in name:
            fm = param
            fm_max = torch.max(fm) 
            fm_min = torch.min(fm)
            ss = torch.div(fm_max-fm_min,quant_level-1) 
            fm_diff = fm - fm_min
            fm_diff_ind = torch.round(torch.div(fm_diff,ss))
            quant_dict[name].copy_(fm_min + fm_diff_ind*ss)
            quant_net_size = quant_net_size + (torch.numel(param)/quant_ratio+2)
            #print(name+' is quantized\n')
        elif 'filter_grid' in name:
            continue
        else:
            quant_dict[name].copy_(param)
            quant_net_size = quant_net_size + torch.numel(param)
    #for i in range(0,len(convs)):
    #    conv = convs[i]
    #    quant_conv = quant_convs[i]
    #    fm = conv.filtermap.data
    #    fm_max = torch.max(fm)
    #    fm_min = torch.min(fm)
    #    ss = torch.div(fm_max-fm_min,255)
    #    fm_diff = fm - fm_min
    #    fm_diff_ind = torch.round(torch.div(fm_diff,ss))
    #    quant_conv.data = fm_min + fm_diff_ind*ss
        #conv.filtermap.data = fm_new
    #    pdb.set_trace()
    return quant_net, quant_net_size

# Model
if os.path.isdir(args.checkpoint_folder_fm):
   ori_net = eval(args.ori_arch + '()')
   #ori_flops, params = get_model_complexity_info(ori_net, (3, 224, 224), cuda = False, as_strings=True, print_per_layer_stat=True)
   #print('Flops of ' + args.ori_arch + ': ' + ori_flops)
   #pdb.set_trace()
   ori_checkpoint = torch.load(args.checkpoint_folder_ori+'/ckpt.t7')
   ori_net = ori_checkpoint['net']
   ori_net_size = net_parameters_size_state_dict(ori_net)
   print('ori_net_size is %d.\n' %(ori_net_size) ) 
   fm_checkpoint = torch.load(args.checkpoint_folder_fm+'/ckpt.t7')
   #fm_checkpoint = torch.load(args.checkpoint_folder_fm+'/ckpt_cycle_0_epoch_99.t7')
   fm_net = fm_checkpoint['net']
   #fm_flops, params = get_model_complexity_info(fm_net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
   #print('Flops of ' + args.fm_arch + ': ' + fm_flops)
   #pdb.set_trace()
   #fmdict = fm_net.state_dict()
   #fmdict_copy = fmdict.copy()
   #for k in fmdict.keys(): 
   #    if 'filter_grid' in k: 
   #        fmdict_copy.pop(k)
   #for name, param in fmdict_copy.items():
   #    if 'filtermap' in name or 'fc' in name:
   #       fmdict_copy[name] = fmdict_copy[name].byte()
   #torch.save(fmdict_copy,'fmdict_copy.pth')
   #fm_net_new = resnet_fm50_imagenet(args.compression_ratio, args.binary_filtermap)
   #fm_net_new.load_state_dict(fmdict_copy)
   #fm_net = fm_net_new
   #pdb.set_trace()
   fm_net_size = net_parameters_size_state_dict(fm_net)
   print('fm_net_size is %d. Begin quantization \n' %(fm_net_size) ) 
   #pdb.set_trace()
   quant_fm_net, quant_fm_net_size = quantization(fm_net, args)
   print('quantization finished, quant_fm_net_size is %d. Begin testing\n' %(quant_fm_net_size) )
   #print_flush_write(args.has_logfile, logfile, 'loading from last saved model, start_epoch : %d, start_cycle : %d\n' \
   #            %(start_epoch, start_cycle))
#pdb.set_trace()
#del fm_net
#torch.cuda.empty_cache()

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
   print('testing the quant_fm_net:\n')
   quant_fm_net = quant_fm_net.to(device)
   if device == 'cuda':
      if args.distributed:
         quant_fm_net = torch.nn.parallel.DistributedDataParallel(quant_fm_net)
      else:
         quant_fm_net = torch.nn.DataParallel(quant_fm_net, device_ids=range(torch.cuda.device_count()))
      cudnn.benchmark = True
   test(0,quant_fm_net)
   quit()
