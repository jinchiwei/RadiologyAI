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

from models import ResNet18_pretrained, inception_v3_pretrained
from dataset import bone_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--network',
    choices=['resnet18'], default='resnet18',
    help='Choose which neural network to use')
args = parser.parse_args()

result_classes = {
  0:'female',
  1:'male'
}



############ testing ############
use_gpu = torch.cuda.is_available()
n_classes = len(result_classes)
if args.network == 'resnet18':
    model = ResNet18_pretrained(n_classes, freeze=False)
    load_file = 'weights_resnet_nonorm/resnet18_weights/005_0.900.pkl'
    val_data_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((256, 256)),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
    ])
elif args.network == 'inception_v3':
    model = inception_v3_pretrained(n_classes, freeze=False)
    load_file = 'weights_inception_nonorm/inception_v3_weights/006_1.000.pkl'
    val_data_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((300, 300)),
      transforms.CenterCrop(299),
      transforms.ToTensor(),
    ])



batch_size=10
model.load_state_dict(torch.load(os.path.join('./', load_file)))
radio_val = bone_dataset(phase='test', transform=val_data_transform)
radio_data_loader = DataLoader(radio_val, batch_size=batch_size, shuffle=True, num_workers=2)

model.train(False)

running_corrects = 0
total = len(radio_val.sample_paths)
print(total)

def do_gpu(x):
  return x.cuda() if use_gpu else x

if use_gpu:
  model = model.cuda()



TP = 0 # pred true, label true
TN = 0 # pred false, label false
FP = 0 # pred true, label false
FN = 0 # pred false, label true

for data in radio_data_loader:
  inputs, ages, labels = data

  """
  # show first images of the batch
  plt.imshow(np.transpose(inputs.numpy()[0], (1,2,0)))
  plt.show()
  """
  # original = inputs
  
  inputs = Variable(do_gpu(inputs)).float()
  ages = Variable(do_gpu(ages)).float()
  labels = Variable(do_gpu(labels)).long()

  # forward
  outputs = model(inputs, ages)
  _, preds = torch.max(outputs.data, 1)

  # statistics
  running_corrects += torch.sum(preds == labels.data)

  # ROC curve analysis
  preds = preds.float().cpu().numpy()
  labels = labels.data.float().cpu().numpy()  
  TP += np.sum(np.logical_and(preds == 1.0, labels == 1.0))
  TN += np.sum(np.logical_and(preds == 0.0, labels == 0.0))
  FP += np.sum(np.logical_and(preds == 1.0, labels == 0.0))
  FN += np.sum(np.logical_and(preds == 0.0, labels == 1.0))

  """
  # show incorrectly classified images
  for idx in range(len(original)):
    if (preds != labels.data)[idx]:
      plt.imshow(np.transpose(original.numpy()[idx], (1,2,0)))
      plt.show()
      # print('here', idx)
  """

print('---------  correct: {:03d} -----------'.format(running_corrects))
print('---------  total: {:03d} -----------'.format(total))
print('---------  accuracy: {:.4f} -----------'.format(running_corrects/total))


sensitivity  = TP / (TP + FN)
specificity  = TN / (TN + FP)
pos_like_ratio = sensitivity / (1 - specificity)
neg_like_ratio = (1 - sensitivity) / specificity
pos_pred_val = TP / (TP + FP)
neg_pred_val = TN / (TN + FN)

print('sensitivity: %f\nspecificity: %f\npositive likelihood value: %f\nnegative likelihood value: %f\npositive predictive value: %f\nnegative predictive value: %f'
        % (sensitivity, specificity, pos_like_ratio, neg_like_ratio, pos_pred_val, neg_pred_val))




