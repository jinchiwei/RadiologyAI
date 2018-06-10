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

# result_classes = {
#   0:'female',
#   1:'male'
# }

label_dir = os.path.join(dataset_dir, csv_labels['global'])
infile_dict = open(label_dir, 'r')
csvreader = csv.DictReader(infile_dict)
global_list = []
for line in csvreader:
	"""
	Image Index
	Finding Labels
	Follow-up #
	Patient ID
	Patient Age
	Patient Gender
	View Position
	OriginalImage[Width
	Height
	OriginalImagePixelSpacing[x
	y
	"""
	if int(line['Patient Age']) <= 100:
		tuple_line = line['Image Index'] + ',' + line['Patient Age'] + ',' + line['Patient Gender']
		global_list.append(tuple_line)
	# print(line['Image Index'])
	# print(line['Patient Age'])
	# print(line['Patient Gender'])
	# break

infile_dict.close()

dataset_size = len(list(global_list))
train_size = int(dataset_size * 0.7)
val_size = int(dataset_size * 0.1)
test_size = int(dataset_size - train_size - val_size)


train_list = global_list[0 : train_size]
val_list = global_list[train_size : train_size + val_size]
test_list = global_list[train_size + val_size :]

header = 'file,age,sex'

thefile = open('chest-training-dataset.csv', 'w')
thefile.write("%s\n" % header)
for item in train_list:
	thefile.write("%s\n" % item)
thefile.close()

thefile = open('chest-val-dataset.csv', 'w')
thefile.write("%s\n" % header)
for item in val_list:
	thefile.write("%s\n" % item)
thefile.close()

thefile = open('chest-test-dataset.csv', 'w')
thefile.write("%s\n" % header)
for item in test_list:
	thefile.write("%s\n" % item)
thefile.close()


"""
[('Image Index', '00000001_000.png'),
	('Finding Labels', 'Cardiomegaly'),
	('Follow-up #', '0'),
	('Patient ID', '1'),
	('Patient Age', '58'),
	('Patient Gender', 'M'),
	('View Position', 'PA'),
	('OriginalImage[Width', '2682'),
	('Height]', '2749'),
	('OriginalImagePixelSpacing[x', '0.143'),
	('y]', '0.143'),
	('', None)]
"""






