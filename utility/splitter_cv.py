from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold, StratifiedKFold
import numpy as np
import os, sys, shutil

'''
This is a custom class that has the property of ImageFolder class in PyTorch, but offers
special utility methods for my own research. It is compatitle with DataLoader in PyTorch
'''
################################################################################################################################
class ImageDataset(ImageFolder):
    # private methods
    def __getitem__(self, index):
        sample, label = super(ImageDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return sample, label, path
    # private method for build_dataset
    def __builddirs__(self, split_dir, train_test_flag):
        if train_test_flag:
            dirs = ['train', 'test']
        else:
            dirs = ['train', 'validation', 'test']
        for dir in dirs:
            current_dir = os.path.join(split_dir, dir)
            if os.path.exists(current_dir):
                print('{} already exists, continue...'.format(os.path.abspath(current_dir)))
            else:
                os.mkdir(current_dir)
                for class_name in self.classes:
                    current_class_dir = os.path.join(current_dir, class_name)
                    if os.path.exists(current_class_dir):
                        print('{} directory already exists in {}'.format(class_name, os.path.abspath(current_dir)))
                    else:
                        os.mkdir(current_class_dir)
                print('{} is successfully created...'.format(os.path.abspath(current_dir)))
        return dirs
    # private method for build_cv_dataset
    def __buildcvdirs__(self, num_splits, split_dir, train_test_flag):
        cv_data_dir = os.path.join(split_dir, 'cross_validation_data')
        if os.path.exists(cv_data_dir):
            print('{} already exists in {}, please check and modify, exiting now...'
                  .format(cv_data_dir, split_dir))
            sys.exit(1)
        else:
            os.mkdir(cv_data_dir)
            if train_test_flag:
                dirs = ['train', 'test']
            else:
                dirs = ['train', 'validation', 'test']
            split_paths = []
            for split_index in range(num_splits):
                split_index = split_index + 1
                current_split_dir = os.path.join(cv_data_dir, 'split_{}_data'.format(split_index))
                if os.path.exists(current_split_dir):
                    print('split_{}_data directly already exists in {}'.format(split_index, os.path.abspath(cv_data_dir)))
                else:
                    os.mkdir(current_split_dir)
                split_paths.append(current_split_dir)
                for dir in dirs:
                    current_dir = os.path.join(current_split_dir, dir)
                    if os.path.exists(current_dir):
                        print('{} directory already exists in {}'.format(dir, os.path.abspath(current_split_dir)))
                    else:
                        os.mkdir(current_dir)
                    for class_name in self.classes:
                        current_class_dir = os.path.join(current_dir, class_name)
                        if os.path.exists(current_class_dir):
                            print('{} directory already exists in {}'.format(class_name, os.path.abspath(current_dir)))
                        else:
                            os.mkdir(current_class_dir)
                print('split_{} directories successfully built...'.format(split_index))
        return split_paths, dirs
    # private method for 4 public get_... methods
    def __getdataset__(self):
        dataset = dict()
        for path, label in self.imgs:
            dataset[path] = label
        return dataset
    # private method for build_cv_dataset
    def __getclass__(self, label):
        for class_name, target in self.class_to_idx.items():
            if target == label:
                return class_name

    # public methods
    def get_train_test_dataset(self, train_ratio, stratified=False, random_seed=0):
        dataset = self.__getdataset__()
        paths = [path for path in dataset.keys()]
        labels = [label for label in dataset.values()]
        if stratified:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=random_seed)
            for train_indices, test_indices in sss.split(paths, labels):
                train_dataset = {paths[train_index] : labels[train_index]
                                 for train_index in train_indices}
                test_dataset = {paths[test_index] : labels[test_index]
                               for test_index in test_indices}
        else:
            ss = ShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=random_seed)
            for train_indices, test_indices in ss.split(paths):
                train_dataset = {paths[train_index] : labels[train_index]
                                 for train_index in train_indices}
                test_dataset = {paths[test_index] : labels[test_index]
                               for test_index in test_indices}
        return train_dataset, test_dataset
    def get_train_val_test_dataset(self, train_ratio, test_ratio, stratified=False, random_seed=0):
        dataset = self.__getdataset__()
        paths = [path for path in dataset.keys()]
        labels = [label for label in dataset.values()]
        indices = np.array(list(range(len(paths))))
        if stratified:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, test_size=test_ratio, random_state=random_seed)
            for train_indices, test_indices in sss.split(paths, labels):
                train_test_indices = np.concatenate((train_indices, test_indices))
                val_indices = np.delete(indices, train_test_indices)
                train_dataset = {paths[train_index] : labels[train_index]
                                 for train_index in train_indices}
                val_dataset = {paths[val_index] : labels[val_index]
                               for val_index in val_indices}
                test_dataset = {paths[test_index] : labels[test_index]
                                for test_index in test_indices}
        else:
            ss = ShuffleSplit(n_splits=1, train_size=train_ratio, test_size=test_ratio, random_state=random_seed)
            for train_indices, test_indices in ss.split(paths):
                train_test_indices = np.concatenate((train_indices, test_indices))
                val_indices = np.delete(indices, train_test_indices)
                train_dataset = {paths[train_index] : labels[train_index]
                                 for train_index in train_indices}
                val_dataset = {paths[val_index] : labels[val_index]
                               for val_index in val_indices}
                test_dataset = {paths[test_index] : labels[test_index]
                                for test_index in test_indices}
        return train_dataset, val_dataset, test_dataset
    def get_train_test_cvdataset(self, num_splits, shuffle=False, stratified=False, random_seed=0):
        dataset = self.__getdataset__()
        paths = [path for path in dataset.keys()]
        labels = [label for label in dataset.values()]
        split_index = 1
        cross_val_data = dict()
        if stratified:
            skf = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=random_seed)
            for train_indices, test_indices in skf.split(paths, labels):
                train_dataset = {paths[train_index] : labels[train_index]
                                 for train_index in train_indices}
                test_dataset = {paths[test_index] : labels[test_index]
                               for test_index in test_indices}
                cross_val_data[split_index] = (train_dataset, test_dataset)
                split_index += 1
        else:
            kf = KFold(n_splits=num_splits, shuffle=shuffle, random_state=random_seed)
            for train_indices, test_indices in kf.split(paths):
                train_dataset = {paths[train_index] : labels[train_index]
                                 for train_index in train_indices}
                test_dataset = {paths[test_index] : labels[test_index]
                               for test_index in test_indices}
                cross_val_data[split_index] = (train_dataset, test_dataset)
                split_index += 1
        return cross_val_data
    def get_train_val_test_cvdataset(self, num_splits, val_ratio, shuffle=False, stratified=False, random_seed=0):
        dataset = self.__getdataset__()
        paths = [path for path in dataset.keys()]
        labels = [label for label in dataset.values()]
        train_ratio = 1 - val_ratio - 1.0 / num_splits
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        split_index = 1
        cross_val_data = dict()
        if stratified:
            skf = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=random_seed)
            for train_val_indices, test_indices in skf.split(paths, labels):
                test_dataset = {paths[test_index] : labels[test_index]
                               for test_index in test_indices}
                train_val_dataset = {paths[train_val_index] : labels[train_val_index]
                                     for train_val_index in train_val_indices}
                train_val_paths = [train_val_path for train_val_path in train_val_dataset.keys()]
                train_val_labels = [train_val_label for train_val_label in train_val_dataset.values()]
                sss = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=random_seed)
                for train_indices, val_indices in sss.split(train_val_paths, train_val_labels):
                    train_dataset = {train_val_paths[train_index] : train_val_labels[train_index]
                                     for train_index in train_indices}
                    val_dataset = {train_val_paths[val_index] : train_val_labels[val_index]
                                   for val_index in val_indices}
                cross_val_data[split_index] = (train_dataset, val_dataset, test_dataset)
                split_index += 1
        else:
            kf = KFold(n_splits=num_splits, shuffle=shuffle, random_state=random_seed)
            for train_val_indices, test_indices in kf.split(paths):
                test_dataset = {paths[test_index] : labels[test_index]
                               for test_index in test_indices}
                train_val_dataset = {paths[train_val_index] : labels[train_val_index]
                                     for train_val_index in train_val_indices}
                train_val_paths = [train_val_path for train_val_path in train_val_dataset.keys()]
                train_val_labels = [train_val_label for train_val_label in train_val_dataset.values()]
                ss = ShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=random_seed)
                for train_indices, val_indices in ss.split(train_val_paths):
                    train_dataset = {train_val_paths[train_index] : train_val_labels[train_index]
                                     for train_index in train_indices}
                    val_dataset = {train_val_paths[val_index] : train_val_labels[val_index]
                                   for val_index in val_indices}
                cross_val_data[split_index] = (train_dataset, val_dataset, test_dataset)
                split_index += 1
        return cross_val_data
    # ratios = [train_ratio] or ratios = [train_ratio, test_ratio] or any iterables
    def build_dataset(self, ratios, dest_dir='.', stratified=False, random_seed=0):
        if len(ratios) == 1:
            train_test_flag = True
        elif len(ratios) == 2:
            train_test_flag = False
        else:
            print('{} has more than 2 ratios, please enter either train_ratio for train/test split, or train_ratio and test_ratio for train/val/test split'
                  .format(ratios))
            sys.exit(1)
        # build data containers/directories
        dirs = self.__builddirs__(split_dir=dest_dir, train_test_flag=train_test_flag)
        if train_test_flag:
            datasets = self.get_train_test_dataset(train_ratio=ratios[0], stratified=stratified, random_seed=random_seed)
        else:
            datasets = self.get_train_val_test_dataset(train_ratio=ratios[0], test_ratio=ratios[1], stratified=stratified, random_seed=random_seed)

        for dataset, dir in zip(datasets, dirs):
            for sample_path, label in dataset.items():
                class_name = self.__getclass__(label)
                copy_to_path = os.path.join(dest_dir, dir, class_name)
                shutil.copy(sample_path, copy_to_path)
            print('{} images have been completely copied...'.format(dir))
    def build_cv_dataset(self, num_splits, val_ratio=0, dest_dir='.', shuffle=False, stratified=False, random_seed=0):
        if val_ratio == 0:
            train_test_flag = True
        else:
            train_test_flag = False
        split_paths, dirs = self.__buildcvdirs__(num_splits=num_splits, split_dir=dest_dir, train_test_flag=train_test_flag)
        if train_test_flag:
            cv_data = self.get_train_test_cvdataset(num_splits, shuffle, stratified, random_seed)
        else:
            cv_data = self.get_train_val_test_cvdataset(num_splits, val_ratio, shuffle, stratified, random_seed)
        for current_split_path, current_split_data in zip(split_paths, cv_data.values()):
            for dir, current_dataset in zip(dirs, current_split_data):
                for path, label in current_dataset.items():
                    class_name = self.__getclass__(label)
                    dest_path = os.path.join(current_split_path, dir, class_name)
                    shutil.copy(path, dest_path)
                print('{} set of {} has successfully completed copying images...'.format(dir, current_split_path))
################################################################################################################################
if __name__ == "__main__":
    data_dir = '/home/vince/programming/python/research/rail/Collaborations_w_Alvin/project2/data'
    melanoma = ImageDataset(data_dir)
    ratios=(0.8, )
    melanoma.build_cv_dataset(5, val_ratio=0.1, dest_dir='academics', shuffle=True, stratified=True)
