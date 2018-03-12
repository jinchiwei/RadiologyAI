# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import matplotlib
matplotlib.use('Agg')
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


def ResNet18_pretrained(n_classes, freeze=True):
  model = models.__dict__['resnet18'](pretrained=True)
  ## freeze all weights
  if freeze:
    for param in model.parameters():
      param.requires_grad = False
  else:
    for param in model.parameters():
      param.requires_grad = True

  ## change the last 1000-fc to n_classes
  num_filters = model.fc.in_features
  model.fc = nn.Linear(num_filters,n_classes)
  return model


############ dataloader ############
dataset_dir = 'dataset/80_20_00'
directories = {'no_train' : 'no_THA_train',
                'yes_train' : 'yes_THA_train',
                'no_test' : 'no_THA_test',
                'yes_test' : 'yes_THA_test',
                'no_eval' : 'no_THA_eval',
                'yes_eval' : 'yes_THA_eval'}

result_classes = {
  0:'no_THA',
  1:'yes_THA'
}

class THADataset(Dataset):
  def __init__(self, train=True, transform=None):
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
    no_THA_dir = os.path.join(dataset_dir, directories['no_eval'])
    yes_THA_dir = os.path.join(dataset_dir, directories['yes_eval'])
    
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



############ testing ############
use_gpu = torch.cuda.is_available()
n_classes = len(list(result_classes.keys()))
model = ResNet18_pretrained(n_classes,freeze=False)
load_file = 'weights_80_20_00/res18_weights/003_1.000.pkl'
batch_size=10
model.load_state_dict(torch.load(os.path.join('./', load_file)))

val_data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[103.6650538, 105.78144487, 107.41949705],
    #                      std=[24.08016139, 22.86971233, 23.37795106]),
  ])
radio_val = THADataset(train=False, transform=val_data_transform)
radio_data_loader = DataLoader(radio_val, batch_size=batch_size, shuffle=True, num_workers=2)

model.train(False)

running_corrects = 0
total = len(radio_val.sample_paths)
print(total)
for data in radio_data_loader:
  inputs, labels = data
  
  if use_gpu:
    inputs = Variable(inputs.cuda()).float()
    labels = Variable(labels.cuda()).long()
  else:
    inputs = Variable(inputs).float()
    labels = Variable(labels).long()
  
  # forward
  outputs = model(inputs)
  _, preds = torch.max(outputs.data, 1)

  # statistics
  running_corrects += torch.sum(preds == labels.data)

print('---------  correct: {:03d} -----------'.format(running_corrects))
print('---------  total: {:03d} -----------'.format(total))
print('---------  accuracy: {:.4f} -----------'.format(running_corrects/total))

