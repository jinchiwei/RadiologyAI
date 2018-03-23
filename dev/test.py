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

from models import ResNet18_pretrained, GoogLeNet_pretrained
from dataset import THADataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--network',
    choices=['googlenet', 'resnet18'], default='resnet18',
    help='Choose which neural network to use')
args = parser.parse_args()

result_classes = {
  0:'no_THA',
  1:'yes_THA'
}

############ testing ############
use_gpu = torch.cuda.is_available()
n_classes = len(result_classes)
if args.network == 'resnet18':
    model = ResNet18_pretrained(n_classes, freeze=False)
    load_file = 'weights_resnet_nonorm/res18_weights/005_0.900.pkl'
elif args.network == 'googlenet':
    model = GoogLeNet_pretrained(n_classes, freeze=False)
    load_file = 'weights_google_nonorm/res18_weights/009_0.950.pkl'

batch_size=10
model.load_state_dict(torch.load(os.path.join('./', load_file)))

val_data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    # transforms.Resize((32, 32)),
    transforms.CenterCrop(224),
    # transforms.CenterCrop(32),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[0.4059296, 0.40955055, 0.412535],
    #                      std=[0.21329397, 0.215493, 0.21677108]),
  ])
radio_val = THADataset(train='test', transform=val_data_transform)
radio_data_loader = DataLoader(radio_val, batch_size=batch_size, shuffle=True, num_workers=2)

model.train(False)

running_corrects = 0
total = len(radio_val.sample_paths)
print(total)

def do_gpu(x):
  return x.cuda() if use_gpu else x

if use_gpu:
  model = model.cuda()

for data in radio_data_loader:
  inputs, labels = data

  # plt.imshow(np.transpose(inputs.numpy()[0], (1,2,0)))
  # plt.show()

  original = inputs
  inputs = Variable(do_gpu(inputs)).float()
  labels = Variable(do_gpu(labels)).long()

  # forward
  outputs = model(inputs)
  _, preds = torch.max(outputs.data, 1)

  # statistics
  running_corrects += torch.sum(preds == labels.data)

  """
  for idx in range(len(original)):
    if (preds != labels.data)[idx]:
      plt.imshow(np.transpose(original.numpy()[idx], (1,2,0)))
      plt.show()
      # print('here', idx)
  """

print('---------  correct: {:03d} -----------'.format(running_corrects))
print('---------  total: {:03d} -----------'.format(total))
print('---------  accuracy: {:.4f} -----------'.format(running_corrects/total))

