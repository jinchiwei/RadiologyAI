from torch.utils.data.dataset import Dataset
import os, sys
from skimage import io
import numpy as np

############ dataloader ############
dataset_dir = 'dataset/100_20_30'
directories = {'no_train' : 'no_THA_train',
                'yes_train' : 'yes_THA_train',
                'no_val' : 'no_THA_val',
                'yes_val' : 'yes_THA_val',
                'no_test' : 'no_THA_test',
                'yes_test' : 'yes_THA_test'}

result_classes = {
  0:'no_THA',
  1:'yes_THA'
}

class THADataset(Dataset):
  def __init__(self, train, transform=None):
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
    if train is 'train':
      no_THA_dir = os.path.join(dataset_dir, directories['no_train'])
      yes_THA_dir = os.path.join(dataset_dir, directories['yes_train'])
    elif train is 'val':
      no_THA_dir = os.path.join(dataset_dir, directories['no_val'])
      yes_THA_dir = os.path.join(dataset_dir, directories['yes_val'])
    else: # train is 'test'
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



if __name__ == '__main__':
  dataset = THADataset('val')
  print(dataset.sample_paths)
