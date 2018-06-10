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



dataset_dir = 'rsna-bone-age/'
images = 'images'
csv_labels = {'train' : 'boneage-training-dataset.csv',
                'val' : 'boneage-val-dataset.csv',
                'test' : 'boneage-test-dataset.csv'}

result_classes = {
  0:'female',
  1:'male'
}
 
phases = ['train', 'val', 'test'] 
count = 0
for phase in phases:
	label_dir = os.path.join(dataset_dir, csv_labels[phase])
	infile_dict = open(label_dir, 'r')
	csvreader = csv.DictReader(infile_dict)
	for line in csvreader: # id,boneage,male
	  age = int(line['boneage'])
	  if age > 12*14:
	  	count += 1
	infile_dict.close()
print(count)






