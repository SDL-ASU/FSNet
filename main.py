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
parser.add_argument('--compression_ratio', '-compression_ratio', default=1, type=float, \
                    help='compression ratio for 1x1 convolution layers')
parser.add_argument('--binary_filtermap', '-bfm', action='store_true', help='using binary filtermap')

#resume
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--load_from_folder', '-l', type=str, default='', help='location of the checkpoint to load model')
parser.add_argument('--resume_last_epoch', action='store_true', help='resume from last epoch')
parser.add_argument('--load_model_pth', type=str, default='', help='location of the model to be loaded')

parser.add_argument('--lr_short_epoch', '-lr_short_epoch', action='store_true', help='using lr policy [82,123]')
parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument('--checkpoint_folder', type=str, default='', help='location of the checkpoint')
parser.add_argument('--transform_net', '-t', action='store_true', help='using transform net')

#arch
parser.add_argument('--arch', '-arch', type=str, default='resnet_fm18', help='network architecture')

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
parser.add_argument('--save_name_suffix', '-save_name_suffix', type=str, default='', help='suffix of the saved model name')
args = parser.parse_args()

args.distributed = args.world_size > 1
if args.distributed:
   dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
start_cycle = 0
resumed = False  #whether the current execution is resumed from a previous one

#log
start_cycle = 0
if args.logfile_name!='':
   logfile_name = args.logfile_name
elif 'fm' in args.arch:
   logfile_name = args.arch+'_compression_'+str(args.compression_ratio)+'.log'
else:
   logfile_name = args.arch+'.log'
logfile = open(logfile_name,"a+")
#if args.has_logfile:
#   if os.path.isfile(logfile_name):
#      logfile = open(logfile_name,"r")
#      #lines = logfile.readlines()
#      text = logfile.read()
#      start_cycles = [int(text[m.end(1):m.end(2)]) for m in \
#                   re.finditer('(cycle )([0-9]+)', text)]
#      if not start_cycles:
#         print('start_cycles is empty in the log \n')
#      else:
#         start_cycle = start_cycles[-1]
#      logfile.close()
#
#logfile = open(logfile_name,"a+")

# Data
print_flush_write(args.has_logfile, logfile, '==> Preparing data..\n')

trainloader, testloader = get_dataloader(args, logfile)
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def valid_net_parameters(net_params):
    for param_iter in self.named_parameters():
        yield param_iter

def net_parameters_size(net):
    #print('Total model parameters: {}'.format(np.sum([np.prod(list(w.size())) for w in model.parameters()]))
    #sum(p.numel() for p in model.parameters() if p.requires_grad)
    #pdb.set_trace()
    parameter_count = 0
    for param in net.parameters():
        #if not isinstance(param.data,torch.cuda.LongTensor): filtermap ids are not registered as parameter
        parameter_count  = parameter_count + torch.numel(param.data)
    return parameter_count

def net_parameters_size_state_dict(net):
    parameter_count = 0
    for name, param in net.state_dict().items():
        parameter_count  = parameter_count + torch.numel(param.data)
    return parameter_count

# Model
if args.resume:
    # Load checkpoint.
    print_flush_write(args.has_logfile, logfile, '==> Resuming from checkpoint..\n')
    if args.load_model_pth != '':
       model_pth = args.load_model_pth
    else:
       load_folder = args.load_from_folder
       if load_folder == '':
          load_folder = args.checkpoint_folder
       model_pth = load_folder+'/ckpt.t7'

    assert os.path.isfile(model_pth), 'Error: did not find  model to be resumed!'
    checkpoint = torch.load(model_pth)
    net = checkpoint['net']
    if args.dataset != "imagenet":
       model_acc = checkpoint['acc']
    else:
       model_acc = checkpoint['top5']
    if args.resume_last_epoch:
       start_epoch = checkpoint['epoch']
    start_cycle = checkpoint['cycle']
    net_size = net_parameters_size_state_dict(net)
    resumed = True
    #if args.transform_net:
    #   if args.opt_compressed_weight:
    #      set_compressed_weight_grad(net,True)
    #      set_transform_grad(net, False)
    #   if args.opt_transformation:
    #      set_transform_grad(net, True)
    #      set_compressed_weight_grad(net, False)
    print_flush_write(args.has_logfile, logfile, \
            'resumed from the acc: %.3f, net_parameters_size is %d\n\n' % (model_acc,net_size))
else:
    print_flush_write(args.has_logfile, logfile, '==> Building model..\n')
    # net = VGG('VGG19')
    if os.path.isdir(args.checkpoint_folder):
       checkpoint = torch.load(args.checkpoint_folder+'/ckpt.t7')
       net = checkpoint['net']
       #pdb.set_trace()
       start_epoch = checkpoint['epoch']
       start_cycle = checkpoint['cycle']
       resumed = True
       print_flush_write(args.has_logfile, logfile, 'loading from last saved model, start_epoch : %d, start_cycle : %d\n' \
               %(start_epoch, start_cycle))
       #pdb.set_trace()
    else:
       net = construct_network(args, logfile)
    
    net_size = net_parameters_size_state_dict(net)
    print_flush_write(args.has_logfile, logfile, 'net_parameters_size is %d\n\n' % (net_size))

net = net.to(device)
if device == 'cuda':
    if args.distributed:
       net = torch.nn.parallel.DistributedDataParallel(net)
    else:
       net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

#net_params = [];
#for param in net.parameters():
#    if param.requires_grad == True:
#       net_params.append(param)
#param_iter = valid_net_parameters(net_params)

criterion = nn.CrossEntropyLoss()
if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
elif args.opt == 'adam':
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
elif args.opt == 'rmsprop':
    optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    print_flush_write(args.has_logfile, logfile, '\nEpoch: %d\n' % epoch)
    net.train()
    train_loss = 0
    train_loss_classify = 0
    train_loss_ref_resnet = 0
    train_loss_fit = 0
    correct = 0
    total = 0
    train_prec1 = 0
    train_prec5 = 0
    num_batches = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)        
        
        optimizer.zero_grad()
        #pdb.set_trace()
        _outputs = net(inputs)
        if isinstance(_outputs, list):
           outputs = _outputs[0]
        else:
           outputs = _outputs
        #pdb.set_trace()
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if args.dataset != "imagenet":
           _, predicted = torch.max(outputs.data, 1)
           total += targets.size(0)
           correct += predicted.eq(targets).sum().item()
        else:
           prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
           train_prec1 += prec1[0]
           train_prec5 += prec5[0]
        if args.progress_bar:
           if args.dataset != "imagenet":
              progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                   % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
           else:
              progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | top-1: %.3f%% | top-5: %.3f%% '
                   % (train_loss/(batch_idx+1), train_prec1/(batch_idx+1), train_prec5/(batch_idx+1)))
        num_batches = num_batches + 1

    if args.dataset != "imagenet":
       print_flush_write(args.has_logfile, logfile, \
               'train_loss: %.3f, train_acc: %.3f%% (%d/%d)\n' %  (train_loss/num_batches, 100.*correct/total, correct, total))
    else:
       print_flush_write(args.has_logfile, logfile, \
               'train_loss: %.3f, top-1: %.3f%%, top-5: %.3f%%\n' % (train_loss/num_batches, train_prec1/num_batches, \
               train_prec5/num_batches))
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


def test(epoch, cycle, net, *save_args):
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

    #if len(save_args) > 0:
    if epoch == args.epoches-1 or epoch%2 == 0 or args.dataset == "imagenet":
    #if acc > best_acc:
        print('Saving..')
        if args.dataset != "imagenet":
           state = {
              'net': net.module if device == 'cuda' else net,
              'acc': acc,
              'epoch': epoch,
              'cycle': cycle,
           }
        else:
           state = {
              'net': net.module if device == 'cuda' else net,
              'top1': top1,
              'top5': top5,
              'epoch': epoch,
              'cycle': cycle,
           }

        save_path_exist = os.path.isdir(args.checkpoint_folder)
        if save_path_exist or args.checkpoint_folder != '':
            if not save_path_exist:
              os.mkdir(args.checkpoint_folder)
            torch.save(state, args.checkpoint_folder+'/ckpt'+args.save_name_suffix+'.t7')
            if epoch == args.epoches-1:
               torch.save(state, args.checkpoint_folder+'/ckpt_cycle_%d_epoch_%d.t7' % (cycle,epoch)) 
        #best_acc = acc


def adjust_filtermap(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        base_lr = args.lr
        #if epoch < 150: lr = 1e-1
        if epoch < 82*2: lr = base_lr
        elif epoch == 82*2: lr = base_lr/10
        elif epoch == 82*3: lr = base_lr/100
        elif epoch == 82*4: lr = base_lr/1000
        #elif epoch == 150: lr = 1e-2
        #elif epoch == 225: lr = 1e-3
        #elif epoch == 250: lr = 1e-3
        else: return
        #lr = 2*lr;
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def adjust_imagenet(optAlg, optimizer, epoch, epoches_stage = 30):
    if optAlg == 'sgd':
        base_lr = args.lr
        #if epoch < 150: lr = 1e-1
        #epoches_stage = args.epoches/3
        #epoches_stage = 30
        if epoch < epoches_stage:
            lr = base_lr
        elif epoch >= epoches_stage and epoch < 2*epoches_stage: 
            lr = base_lr/10
        elif epoch >= 2*epoches_stage and epoch < 3*epoches_stage: lr = base_lr/100
        elif epoch >= 3*epoches_stage: lr = base_lr/1000
        #elif epoch == 150: lr = 1e-2
        #elif epoch == 225: lr = 1e-3
        #elif epoch == 250: lr = 1e-3
        else: return
        #lr = 2*lr;
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def adjust_filtermap_clr(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        base_lr = args.lr
        #if epoch < 150: lr = 1e-1
        #if epoch < 82*2: lr = base_lr
        #elif epoch == 82*2: lr = base_lr/10
        #elif epoch == 82*3: lr = base_lr/100
        #elif epoch == 82*4: lr = base_lr/1000
        x1 = 82*2
        x2 = 82*3
        x3 = 82*4
        x4 = 409

        cycle = math.floor(epoch/(2*x4))
        cycle_idx = math.floor(epoch/x4) - 2*cycle
        if cycle_idx == 0:
           epoch_idx = epoch - cycle*2*x4
           if epoch_idx < x4-x3+1: lr = base_lr/1000
           elif epoch_idx == x4-x3+1: lr = base_lr/10
           elif epoch_idx == x4-x2+1: lr = base_lr/10
           elif epoch_idx == x4-x1+1: lr = base_lr
           else: return
        else:
           assert cycle_idx == 1
           epoch_idx = epoch - cycle*2*x4 - x4
           if epoch_idx < x1: lr = base_lr
           elif epoch_idx == x1: lr = base_lr/10
           elif epoch_idx == x2: lr = base_lr/100
           elif epoch_idx == x3: lr = base_lr/1000
           else: return
        #lr = 2*lr;
        print("adjust_filtermap_clr: cycle=%d, cycle_idx=%d, lr=%.3f" % (cycle,cycle_idx,lr))

def adjust_opt_general(optAlg, _optimizer, epoch):
    global rcc_loss_lambda
    if optAlg == 'sgd':
        base_lr = args.lr
        lr_point1 = math.floor(args.epoches*0.5)
        lr_point2 = math.floor(args.epoches*0.75)
        if epoch < lr_point1:
           lr = base_lr
        elif epoch >= lr_point1 and epoch < lr_point2:
           lr = base_lr/10
        elif epoch >= lr_point2:
           lr = base_lr/100
        for param_group in _optimizer.param_groups:
            param_group['lr'] = lr

if args.evaluate:
   test(0,net)
   quit()
max_cycle = args.cycle
for cycle in range(int(start_cycle), int(max_cycle)):
    print_flush_write(args.has_logfile, logfile, 'cycle %d:\n' %(cycle))
    if resumed:
        resumed = False
    else:
        start_epoch = 0
    for epoch in range(start_epoch, args.epoches):
        if args.distributed:
           train_sampler.set_epoch(epoch)
        if args.dataset == 'imagenet': #and args.resnet_filtermap_net:
           adjust_imagenet(args.opt, optimizer, epoch, args.epoches_stage)
        else:
           adjust_opt_general(args.opt, optimizer, epoch)
        train(epoch)
        with torch.no_grad():
             test(epoch,cycle,net)
        logfile.flush()
        os.fsync(logfile.fileno())

print_flush_write(args.has_logfile, logfile, 'finished')
if args.has_logfile:
   logfile.close()
