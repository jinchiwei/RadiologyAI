from torch.utils.data.dataset import Dataset
import os, sys
from skimage import io
import numpy as np
import csv
from matplotlib import pyplot as plt
import cv2

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
    # x = io.imread(img_path) # x = io.imread(img_path)[:,:,:3]
    # x = np.resize(x, (x.shape[0], x.shape[1], 3))
    x = cv2.imread(img_path, 1)
    if self.transform:
      x = self.transform(x)
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
  dataset = bone_dataset('test')
  print(dataset.sample_paths)
