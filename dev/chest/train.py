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
from torchvision import datasets, transforms
import torch.optim.lr_scheduler as sch
import torch.nn.functional as F

import os, sys
from skimage import io

from PIL import Image
import time

from models import ResNet18_pretrained
from dataset import chest_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--network',
    choices=['resnet18'], default='resnet18',
    help='Choose which neural network to use')
args = parser.parse_args()

# result_classes = {
#   0:'female',
#   1:'male'
# }



############ training ############
## LOG
weight_out_dir = (args.network + '_weights')
use_gpu = torch.cuda.is_available()

## TRAIN PARAMS
# n_classes = len(list(result_classes.keys()))
n_classes = 4 # or 5
L2_weight_decay = 1e-5
batch_size = 10
num_epochs = 50
lr = 0.001
momentum = 0.9

class_weights = None
if use_gpu and class_weights is not None:
  class_weights = class_weights.cuda().float()

def main():
  ## MODEL
  if args.network == 'resnet18':
    model = ResNet18_pretrained(n_classes,freeze=False)
    print('model is resnet18')
  elif args.network == 'inception_v3':
    model = inception_v3_pretrained(n_classes,freeze=False)
    print('model is inception_v3')
    
  ## LOSS PARAMETER
  criterion = nn.CrossEntropyLoss(weight=class_weights) # equivalent to NLL Loss + softmax = cross entropy
  print("Starting!")
  if use_gpu:
    print("using gpu")
    model = model.cuda()
    criterion = criterion.cuda()
  sys.stdout.flush()
  
  # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=L2_weight_decay)
  optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum, weight_decay=L2_weight_decay)
  # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=L2_weight_decay)
  # optimizer = torch.optim.Adam(model.fc.parameters(),lr=lr,weight_decay=L2_weight_decay)
  # optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=L2_weight_decay)
  exp_lr_scheduler = sch.StepLR(optimizer, step_size=50, gamma=0.1)

  model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)
  

def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
  if args.network == 'resnet18':
    train_data_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((256, 256)),
      transforms.RandomRotation((-5, +5)),
      transforms.RandomCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
    ])
    val_data_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((256, 256)),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
    ])
  elif args.network == 'inception_v3':
    train_data_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((300, 300)),
      transforms.RandomRotation((-5, +5)),
      transforms.RandomCrop(299),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
    ])
    val_data_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((300, 300)),
      transforms.CenterCrop(299),
      transforms.ToTensor(),
    ])
  
  radio_train = chest_dataset(phase='train', transform=train_data_transform)
  radio_val = chest_dataset(phase='val', transform=val_data_transform)

  dataloaders = {
    'train': DataLoader(radio_train, batch_size=batch_size, shuffle=True, num_workers=2),
    'val': DataLoader(radio_val, batch_size=batch_size, shuffle=True, num_workers=2)
  }

  ##### TRAIN ROUTINE
  since = time.time()
  best_model_wts = model.state_dict()
  best_acc = 0.0

  if not os.path.exists('weights/' + weight_out_dir):
    os.makedirs('weights/' + weight_out_dir)
  LOG_FILE = open('weights/' + weight_out_dir + '/LOG.txt', 'w')

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)


    epoch_info = [0] * 4
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        scheduler.step()
        model.train(True)  # Set model to training mode
      else:
        model.train(False)  # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      count = 0
      dataset_size = len(dataloaders[phase])
      
      for data in dataloaders[phase]:
        # get the inputs
        inputs, ages, sexes = data
        if use_gpu:
          inputs = Variable(inputs.cuda()).float()
          ages = Variable(ages.cuda()).long()
          sexes = Variable(sexes.cuda()).float()
        else:
          inputs = Variable(inputs).float()
          ages = Variable(ages).long()
          sexes = Variable(sexes).float()
        
        # increment the count
        count += 1

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs, sexes)

        # for nets that have multiple outputs such as inception
        if isinstance(outputs, tuple):
          _, preds = torch.max(outputs[0].data, 1)
          loss = sum((criterion(o,ages) for o in outputs))
        else:
          _, preds = torch.max(outputs.data, 1)
          loss = criterion(outputs, ages)

        # backward + optimize only if in training phase
        if phase == 'train':
          loss.backward()
          optimizer.step()

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == ages.data)
        # print('preds: ', preds)
        # print('ages: ', ages.data)

        print('{:d}/{:d}:  {:s}_loss: {:.3f}, {:s}_acc: {:.3f} \r'.format(batch_size*count,
                                                                          batch_size*dataset_size,
                                                                          phase,
                                                                          running_loss / count,
                                                                          phase,
                                                                          running_corrects / (count * batch_size)),
              end='\r')
        sys.stdout.flush()

      epoch_loss = running_loss / dataset_size
      epoch_acc = running_corrects / (count * batch_size)

      print('---------  {} Loss: {:.4f} Acc: {:.4f} -----------'.format(phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = model.state_dict()
        torch.save(best_model_wts, 'weights/' + weight_out_dir + '/{:03d}_{:.3f}.pkl'.format(epoch, epoch_acc))
      if phase == 'train':
        epoch_info[0] = epoch_loss
        epoch_info[1] = epoch_acc
      else:
        epoch_info[2] = epoch_loss
        epoch_info[3] = epoch_acc
    LOG_FILE.write(str(epoch_info[0]) + ' ' + str(epoch_info[1]) + ' ' + str(epoch_info[2])
                   + ' ' + str(epoch_info[3]) + '\n')
    print()
  LOG_FILE.close()
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model



if __name__ == '__main__':
  main()

