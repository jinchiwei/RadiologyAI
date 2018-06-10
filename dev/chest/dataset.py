import torch
from torch.utils.data.dataset import Dataset
import os, sys
from skimage import io, color
import numpy as np
import csv
from matplotlib import pyplot as plt

############ dataloader ############
dataset_dir = 'ChestX-ray8/'
images = 'images'
csv_labels = {'train' : 'chest-training-dataset.csv',
                'val' : 'chest-val-dataset.csv',
                'test' : 'chest-test-dataset.csv'}

result_classes = {
  0:'female',
  1:'male'
}

class chest_dataset(Dataset):
  def __init__(self, phase, transform=None):
    """
    Args:
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    self.transform = transform
    self.sample_paths = []
    self._init(phase)

  def __len__(self):
    return len(self.sample_paths)

  def __getitem__(self, idx):
    img_path, age, gender = self.sample_paths[idx]
    
    x = io.imread(img_path)
    # in order to make everything in RGB (x, y, 3) dimension
    shape = x.shape
    if len(shape) == 3 and shape[2] > 3: # for cases (x, y, 4)
      x = x[:,:,:3]
    elif len(shape) == 2: # for cases (x, y)
      x = color.gray2rgb(x)

    if self.transform:
      x = self.transform(x)
    
    # age = (age) / float(228) # normalize so that we have age := age / max_age, thus 0 <= age <= 1
    # age = age // 10 # so that we have age in decades
    """
    4 bucket: 0-18, 18-40, 40-65, 65-100: 0 1 2 3
    5 bucket: 0-11, 11-18, 18-40, 40-65, 65-100: 0 1 2 3 4
    """
    if age in range(0, 19): # 0-18
      age = 0
    elif age in range(19, 41): # 19-40
      age = 1
    elif age in range(41, 66): # 41-65
      age = 2
    elif age in range(66, 101): # 66-100
      age = 3
    else:
      print(age + 'wtf')

    gender = torch.FloatTensor([gender])

    # print(age)
    # print(gender)
    return (x, age, gender)

  def _init(self, phase):
    image_dir = os.path.join(dataset_dir, images)
    label_dir = os.path.join(dataset_dir, csv_labels[phase])
    infile_dict = open(label_dir, 'r')
    csvreader = csv.DictReader(infile_dict)
    for line in csvreader: # id,boneage,male
      img_path = os.path.join(image_dir, (line['file']))
      age = int(line['age'])
      gender = 0 if line['sex'] == 'F' else 1
      if age <= 100:
        self.sample_paths.append((img_path, age, gender))
    infile_dict.close()



if __name__ == '__main__':
  # print('Test codes are commented out')
  
  dataset = chest_dataset('test')
  # dataset = chest_dataset('val')
  # dataset = chest_dataset('train')

  # print(dataset)
  # print(dataset.__getitem__(0)[0].shape)
  # print(dataset.__getitem__(0)[1])
  # print(dataset.__getitem__(0)[2])
  
  for idx in range(dataset.__len__()):
    dataset.__getitem__(idx)
    # print(dataset.__getitem__(idx)[0].shape)
    # print(dataset.__getitem__(idx)[1])
    # print(dataset.__getitem__(idx)[2])
  print(dataset.__len__())


