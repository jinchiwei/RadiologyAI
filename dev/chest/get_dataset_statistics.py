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

import csv



dataset_dir = 'ChestX-ray8/'
images = 'images'
csv_labels = {'train' : 'chest-training-dataset.csv',
                'val' : 'chest-val-dataset.csv',
                'test' : 'chest-test-dataset.csv',
                'global' : 'Data_Entry_2017.csv'}
  
label_dir = os.path.join(dataset_dir, csv_labels['global'])
infile_dict = open(label_dir, 'r')
csvreader = csv.DictReader(infile_dict)
max = -1
for line in csvreader: # id,boneage,male
  age = int(line['Patient Age'])
  if age > max:
    max = age
    print(max)
print(max)
infile_dict.close()






