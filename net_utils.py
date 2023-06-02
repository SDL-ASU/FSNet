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
import sys
import argparse
import io
import lmdb

import torch
from torchvision import datasets
from PIL import Image

from models import *
from utils import print_flush_write
import pdb


def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def lmdb_imagefolder(root, transform=None, target_transform=None,
                     loader=lmdb_loader):
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')
    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        # print('Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(
            root, transform, target_transform, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        print('Saving pt to {}'.format(pt_path))
        print('Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for path, class_index in data_set.imgs:
                with open(path, 'rb') as f:
                    data = f.read()
                txn.put(path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)
    return data_set


def construct_network(args, logfile):
    if args.transform_net:
       net = Transform_ResNet18(ref_resnet)
       print_flush_write('using Transform_ResNet18\n')
    else:
       arch = args.arch
       #m = re.search('[_a-zA-Z]+([0-9]+)',arch)
       #nlayer = arch[m.start(1):m.end(1)]
       compression_ratio = args.compression_ratio
       binary_filtermap = args.binary_filtermap
       try:
           if 'fm' in arch:
               if args.dataset == "imagenet":
                  net = eval(arch + '(compression_ratio, binary_filtermap)')
               else:
                  net = eval(arch + '(compression_ratio, binary_filtermap)')
           else:
               net = eval(arch + '()')
       except:
           net = resnet18()
           arch = 'resnet18'
           print_flush_write(args.has_logfile, logfile, 'Unexpected error, \
                       you may specify undefined resnet_filtermap_net type, \
                       using resnet18 as the default net now\n' + 'sys.exc_info()[0]:\n' + str(sys.exc_info()[0]) + '\n')
       print_flush_write(args.has_logfile, logfile, 'using ' + arch + '\n')

       #if args.init_filtermap:
       #   ref_resnet_checkpoint = torch.load('./'+args.load_from_folder+'/ckpt.t7')
       #   ref_resnet = ref_resnet_checkpoint['net']
       #   #pdb.set_trace()
       #   net = init_filtermap_net(net,ref_resnet)
       #   print_flush_write('Initialize resnet_fm18 Done')
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    return net

def get_dataloader(args, logfile):

    if args.dataset == 'cifar10':
       transform_train = transforms.Compose([
       transforms.RandomCrop(32, padding=4),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
       ])

       transform_test = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
       ])

       trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
       testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

       trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers)
       testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)

    elif args.dataset == 'svhn':
       def target_transform(target):
           return int(target[0]) - 1
       transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
       transform_test = transform_train
       trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train, \
                 target_transform=target_transform)
       testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test, \
                 target_transform=target_transform)
       trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers)
       testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=args.workers)

    elif args.dataset == 'imagenet':
       # traindir = os.path.join(args.imagenet, 'train')
       # valdir = os.path.join(args.imagenet, 'val')
       # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     # std=[0.229, 0.224, 0.225])

       # train_dataset = datasets.ImageFolder(
        # traindir,
        # transforms.Compose([
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize,
       # ]))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        crop_scale = 0.08
        jitter_param = 0.4
        hue_param = 0.1
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param, hue=hue_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        train_dataset = lmdb_imagefolder(
           # os.path.join(FLAGS.dataset_dir, 'train'),
           '/work/data/image/imagenet/imagenet_lmdb/train',
           transform=train_transforms)
        val_dataset = lmdb_imagefolder(
           # os.path.join(FLAGS.dataset_dir, 'val'),
           '/work/data/image/imagenet/imagenet_lmdb/val',
           transform=val_transforms)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

       # val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # normalize,
       # ]))

        trainloader = torch.utils.data.DataLoader(
           train_dataset, batch_size=args.batchsize, shuffle=(train_sampler is None),
           num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        testloader = torch.utils.data.DataLoader(
           val_dataset, batch_size=200, shuffle=False,
           num_workers=args.workers, pin_memory=True)
    else:
        print_flush_write(args.has_logfile, logfile, 'undefined dataset ' + args.dataset + '\n')
        trainloader = NULL
        testloader = NULL

    return trainloader, testloader
