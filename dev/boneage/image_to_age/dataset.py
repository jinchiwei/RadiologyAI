import torch
from torch.utils.data.dataset import Dataset
import os, sys
from skimage import io, color
import numpy as np
import csv
from matplotlib import pyplot as plt

############ dataloader ############
dataset_dir = 'rsna-bone-age/'
images = 'images'
csv_labels = {'train' : 'boneage-training-dataset.csv',
                'val' : 'boneage-val-dataset.csv',
                'test' : 'boneage-test-dataset.csv'}

result_classes = {
  0:'female',
  1:'male'
}

class bone_dataset(Dataset):
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
    img_path, age, label = self.sample_paths[idx]
    
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
    age = age // 12 # so that we have 20 buckets: 0, 1, ..., 19

    return (x, age, label)

  def _init(self, phase):
    image_dir = os.path.join(dataset_dir, images)
    label_dir = os.path.join(dataset_dir, csv_labels[phase])
    infile_dict = open(label_dir, 'r')
    csvreader = csv.DictReader(infile_dict)
    for line in csvreader: # id,boneage,male
      img_path = os.path.join(image_dir, (line['id'] + '.png'))
      age = int(line['boneage'])
      label = 0 if line['male'] == 'False' else 1
      self.sample_paths.append((img_path, age, label))
    infile_dict.close()



if __name__ == '__main__':
  print('Test codes are commented out')
  
  # dataset = bone_dataset('test')
  # # print(dataset.__getitem__(0)[0].shape)
  # # print(dataset.__getitem__(0)[1])
  # # print(dataset.__getitem__(0)[2])

  # # dataset = bone_dataset('val')
  # # dataset = bone_dataset('train')
  # for idx in range(dataset.__len__()):
  #   dataset.__getitem__(idx)
  #   # dataset.__getitem__(idx)
  #   # print(dataset.__getitem__(idx)[0].shape)
  #   # print(dataset.__getitem__(idx)[1])
  #   # print(dataset.__getitem__(idx)[2])
  # print(dataset.__len__())


