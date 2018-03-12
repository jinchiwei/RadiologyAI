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

# from models import *
# from data import *
# from checkpoints import *

from PIL import Image
import time



############ models ############
# def Alexnet_pretrained(n_classes, freeze=True):
#   ## get the pretrained model
#   model = models.alexnet(pretrained=True)
  
#   ## freeze all weights
#   if freeze:
#     for param in model.parameters():
#       param.requires_grad = False
#   else:
#     for param in model.parameters():
#       param.requires_grad = True

#   model.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, n_classes),
#         )
  
#   return model

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
    no_THA_dir = ''
    yes_THA_dir = ''
    if train:
      no_THA_dir = os.path.join(dataset_dir, directories['no_train'])
      yes_THA_dir = os.path.join(dataset_dir, directories['yes_train'])
    else:
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



############ training ############
## LOG
weight_out_dir = 'res18_weights'
use_gpu = torch.cuda.is_available()

## TRAIN PARAMS
n_classes = len(list(result_classes.keys()))
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
  model = ResNet18_pretrained(n_classes,freeze=False)
  # model = Alexnet_pretrained(n_classes,freeze=False)
  
  ## LOSS PARAMETER
  criterion = nn.CrossEntropyLoss(weight=class_weights) # equivalent to NLL Loss + softmax = cross entropy
  print("Starting!")
  if use_gpu:
    print("using gpu")
    model = model.cuda()ofrr
    criterion = criterion.cuda()
  sys.stdout.flush()

  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=L2_weight_decay)
  # optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum, weight_decay=L2_weight_decay)
  # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=L2_weight_decay)
  # optimizer = torch.optim.Adam(model.fc.parameters(),lr=lr,weight_decay=L2_weight_decay)
  # optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=L2_weight_decay)
  exp_lr_scheduler = sch.StepLR(optimizer, step_size=50, gamma=0.1)

  model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)



def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
  train_data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[103.6650538, 105.78144487, 107.41949705],
    #                      std=[24.08016139, 22.86971233, 23.37795106]),
  ])
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
  
  radio_train = THADataset(train=True, transform=train_data_transform)
  radio_val = THADataset(train=False, transform=val_data_transform)

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
        inputs, labels = data
        if use_gpu:
          inputs = Variable(inputs.cuda()).float()
          labels = Variable(labels.cuda()).long()
        else:
          inputs = Variable(inputs).float()
          labels = Variable(labels).long()
        
        # increment the count
        count += 1

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        if phase == 'train':
          loss.backward()
          optimizer.step()

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

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

