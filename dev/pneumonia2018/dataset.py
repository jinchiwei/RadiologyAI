from torch.utils.data.dataset import Dataset
import os, sys
from skimage import io, color
import numpy as np
import random
import torchvision
import torch
import cv2

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# from matplotlib import pyplot as plt

############ dataloader ############
result_classes = {
    0: 'no_THA',
    1: 'yes_THA',
}

dataset_dir = 'data/'
directories = {}
for class_num in result_classes:
    directories['train_' + str(class_num)] = result_classes[class_num] + '_train'
    directories['val_' + str(class_num)] = result_classes[class_num] + '_val'
    directories['test_' + str(class_num)] = result_classes[class_num] + '_test'

class read_dataset(Dataset):
    def __init__(self, mode, transform=None):
        """
        Args:
           transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.sample_paths = []
        self.sample_paths_mix = []        
        self._init(mode)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        img_path,label = self.sample_paths[idx]
        x = io.imread(img_path)

        # in order to make everything in RGB (x, y, 3) dimension
        shape = x.shape
        if len(shape) == 3 and shape[2] > 3: # for cases (x, y, 4)
            x = x[:,:,:3]
        elif len(shape) == 2: # for cases (x, y)
            x = color.gray2rgb(x)

        # in order to make sure we have images in uint8
        if x.dtype != np.uint8:
            x = x.astype(np.uint8)
        if self.transform:
            x = self.transform(x)

        img_path_mix, label_mix = self.sample_paths_mix[idx]
        x_mix = io.imread(img_path_mix)
        shape = x_mix.shape
        if len(shape) == 3 and shape[2] > 3: # for cases (x, y, 4)
            x_mix = x_mix[:,:,:3]
        elif len(shape) == 2: # for cases (x, y)
            x_mix = color.gray2rgb(x_mix)

        #in order to make sure we have images in uint8
        if x_mix.dtype != np.uint8:
            x_mix = x_mix.astype(np.uint8)
        if self.transform:
            x_mix = self.transform(x_mix)
        
        return (x, label, x_mix, label_mix)

    def _init(self, mode):
        subdir = {}
        # Result class iteration
        for class_num in result_classes:
            subdir[class_num] = os.path.join(dataset_dir, directories[mode + '_' + str(class_num)])
            samples = os.listdir(subdir[class_num])
            for sample in samples:
                if not sample.startswith('.'):  # avoid .DS_Store
                    img_path = os.path.join(subdir[class_num], sample)
                self.sample_paths.append((img_path, class_num))
        # This is to ultimately make sure it is trained w/o order
        random.shuffle(self.sample_paths)

        f = open("data/mix_images_" + mode + ".txt", "rt")
        for line in f:
            copyLine = line
            label_mix = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            img_path_mix = "data/mix_images/" + copyLine[0: 16]
            copyLine = copyLine[17:]
            for x in range(0, 14):
                if copyLine[0] == "1":
                    label_mix[x] = 1
                copyLine = copyLine[2:]
            label_tensor = torch.tensor(label_mix)
            self.sample_paths_mix.append((img_path_mix, label_tensor))

if __name__ == '__main__':
    print("Test code commented out")