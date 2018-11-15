from torch.utils.data.dataset import Dataset
import os, sys
from skimage import io, color
import numpy as np
# from matplotlib import pyplot as plt

############ dataloader ############
result_classes = {
    0: 'no_THA',
    1: 'yes_THA',
    # 2: 'yes_HRA'
}

dataset_dir = 'dataset/100_20_30'
directories = {}
for class_num in result_classes:
    directories['train_' + str(class_num)] = result_classes[class_num] + '_train'
    directories['val_' + str(class_num)] = result_classes[class_num] + '_val'
    directories['test_' + str(class_num)] = result_classes[class_num] + '_test'

# directories = {
#     'train_0' : 'no_THA_train',
#     'train_1' : 'yes_THA_train',
#     'val_0' : 'no_THA_val',
#     'val_1' : 'yes_THA_val',
#     'test_0' : 'no_THA_test',
#     'test_1' : 'yes_THA_test'
# }


class read_dataset(Dataset):
    def __init__(self, mode, transform=None):
        """
        Args:
           transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.sample_paths = []
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
    
        return (x,label)

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


if __name__ == '__main__':
    print('Test codes are commented out')
    # dataset = THADataset('test')
    # dataset = THADataset('val')
    # dataset = THADataset('train')
    # for idx in range(dataset.__len__()):
    # print(idx)
    # x, label = dataset.__getitem__(idx)
    # print(type(x))
    # print(x.dtype)
    # if x.dtype == np.uint16:
    #   print(x.dtype)
    #   x = x.astype(np.uint8)
    #   print(x.dtype)
    # print(x)
