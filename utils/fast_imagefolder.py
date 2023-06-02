import io
import os
import lmdb

import torch
from torchvision import datasets
from PIL import Image



def bytedata_loader(bytedata):
    # In-memory binary streams
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def pickle_imagefolder(
        root, transform=None, target_transform=None, loader=bytedata_loader):
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_fast_imagefolder.pt')
    if os.path.isfile(pt_path):
        # print('Loading pt from {}'.format(pt_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(
            root, transform, target_transform, loader)
        data_set_samples = []
        for path, class_index in data_set.imgs:
            with open(path, 'rb') as f:
                data = f.read()
            data_set_samples.append((data, class_index))
        data_set.imgs_replica = data_set.imgs.copy()
        data_set.imgs = data_set_samples
        # print('Saving pt to {}'.format(pt_path))
        torch.save(data_set, pt_path, pickle_protocol=4)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    return data_set


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


def lmdb_raw_loader(path, lmdb_data, size_data):
    # In-memory binary streams
    with lmdb_data.begin(write=False) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.frombytes(mode='RGB', size=size_data[path], data=bytedata)
    return img


def lmdb_raw_imagefolder(root, transform=None, target_transform=None,
                         loader=lmdb_raw_loader):
    if root.endswith('/'):
        root = root[:-1]
    basename = os.path.basename(root)
    pt_path = os.path.join(
        root + '_lmdb_imagefolder.pt'.format(basename))
    lmdb_path = os.path.join(
        root + '_lmdb_imagefolder.db'.format(basename))
    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        # print('Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(
            root, transform, target_transform, None)
        print('Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        size_data = {}
        with env.begin(write=True) as txn:
            for path, class_index in data_set.imgs:
                with open(path, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                size_data[path] = img.size
                txn.put(path.encode('ascii'), img.tobytes())
        data_set.size_data = size_data
        torch.save(data_set, pt_path, pickle_protocol=4)
        print('Saving pt to {}'.format(pt_path))
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(
        path, data_set.lmdb_data, data_set.size_data)
    return data_set
