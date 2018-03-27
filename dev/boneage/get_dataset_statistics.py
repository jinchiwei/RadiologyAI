# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, models
import torch.optim.lr_scheduler as sch
import torch.nn.functional as F

import os, sys
from skimage import io

from PIL import Image
import time



############ dataloader ############
dataset_dir = 'dataset/100_20_30'
directories = {'no_train' : 'no_THA_train',
                'yes_train' : 'yes_THA_train',
                'no_val' : 'no_THA_val',
                'yes_val' : 'yes_THA_val',
                'no_test' : 'no_THA_test',
                'yes_test' : 'yes_THA_test'}

result_classes = {
  0:'no_THA',
  1:'yes_THA'
}

class THADataset(Dataset):
  def __init__(self, train, transform=None):
    """
    Args:
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    self.transform = transform
    self.sample_paths = []
    self._init(train)

  def __len__(self):
    return len(self.sample_paths)

  def __getitem__(self, idx):
    img_path,label = self.sample_paths[idx]
    x = io.imread(img_path) # x = io.imread(img_path)[:,:,:3]
    x = np.resize(x, (x.shape[0], x.shape[1], 3))
    if self.transform:
      x = self.transform(x)
    
    return (x,label)

  def _init(self, train):
    no_THA_dir = ''
    yes_THA_dir = ''
    if train is 'train':
      no_THA_dir = os.path.join(dataset_dir, directories['no_train'])
      yes_THA_dir = os.path.join(dataset_dir, directories['yes_train'])
    elif train is 'val':
      no_THA_dir = os.path.join(dataset_dir, directories['no_val'])
      yes_THA_dir = os.path.join(dataset_dir, directories['yes_val'])
    else: # train is 'test'
      no_THA_dir = os.path.join(dataset_dir, directories['no_test'])
      yes_THA_dir = os.path.join(dataset_dir, directories['yes_test'])

    
    # NO  
    samples = os.listdir(no_THA_dir)
    for sample in samples:
        if not sample.startswith('.'): # avoid .DS_Store
            img_path = os.path.join(no_THA_dir, sample)
            self.sample_paths.append((img_path,0))
    # YES
    samples = os.listdir(yes_THA_dir)
    for sample in samples:
        if not sample.startswith('.'): # avoid .DS_Store
            img_path = os.path.join(yes_THA_dir, sample)
            self.sample_paths.append((img_path, 1))


batch_size = 100
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
  ])
dset = THADataset(train='train', transform=data_transform)
dataloader = DataLoader(dset, batch_size=batch_size)
print(len(dset.sample_paths)) # 100

for images, labels in dataloader:
    numpy_images = images.numpy()

    per_image_mean = np.mean(numpy_images, axis=(2,3)) # Shape (32,3)
    per_image_std = np.std(numpy_images, axis=(2,3)) # Shape (32,3)

    pop_channel_mean = np.mean(per_image_mean, axis=0) # Shape (3,)
    pop_channel_std = np.mean(per_image_std, axis=0) # Shape (3,)

    print('mean: ', pop_channel_mean)
    print('std: ', pop_channel_std)


