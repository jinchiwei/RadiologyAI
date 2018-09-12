import os
import numpy as np
import cv2 as cv
from sklearn.model_selection import KFold, StratifiedKFold



# input data path
OD_DATA_PATH = "/home/vince/programming/python/rail/data/OD"
OS_DATA_PATH = "/home/vince/programming/python/rail/data/OS"
# parameters for spiltters
NUM_OF_SPLITS = 5



'''
this method returns a dictionary that contains all OD and OS images;
in particular, the keys are filepaths and the values are labels;
each OD image is labeled as 0 while each OS image is labeled as 1;
'''
###################################################################################################
def build_data_dict():
    data_dict = dict()
    # os.fsencode() is used to encode the file directory, otherwise some weird bug occurs
    od_directory = os.fsencode(OD_DATA_PATH)
    os_directory = os.fsencode(OS_DATA_PATH)

    # label and add OD images
    for filename in os.listdir(od_directory):
        file = os.path.join(od_directory, filename)
        if os.path.isfile(file):
            data_dict[file] = 0
    # label and add OS images
    for filename in os.listdir(os_directory):
        file = os.path.join(os_directory, filename)
        if os.path.isfile(file):
            data_dict[file] = 1

    return data_dict
###################################################################################################

'''
this method takes a data dictionary in the format {filename : label}
and outputs two numpy arrays, the first array is numpy representation
of each image and the seocnd array is np.array(label_list);

the two returned numpy arrays are needed for the stratified_kfold and
kfold methods;
'''
###################################################################################################
def build_data_arrays(data_dict):
    files = [filepath.decode('unicode_escape') for filepath in data_dict.keys()]
    images = [cv.imread(file) for file in files]
    labels = [label for label in data_dict.values()]

    return np.array(images), np.array(labels)
###################################################################################################

'''
this method takes two numpy arrays, namely, samples and labels, as well
as three parameter (control the split), and performs the stratified kth
fold split;

each entry in the returned data_dict has the following form
    index : (x_train, x_test, y_train, y_test)
where index = i represents the ith split
'''
###################################################################################################
def stratified_kfold(samples, labels, num_of_splits, shuffle_state=True, random_num=123):
    skf = StratifiedKFold(n_splits=num_of_splits, shuffle=shuffle_state, random_state=random_num)
    data_dict = dict()
    index = 1
    for train_index, test_index in skf.split(samples, labels):
        x_train, x_test = samples[train_index], samples[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        data_dict[index] = (x_train, x_test, y_train, y_test)
        index += 1
    return data_dict
###################################################################################################

'''
the kfold method is very similar to stratified_kfold, the only difference
is that the kfold method does not ensure an approximately equal split, that
is, the labels may be highly unbalanced;
'''
###################################################################################################
def kfold(samples, labels, num_of_splits, shuffle_state=True, random_num=123):
    kf = KFold(n_splits=num_of_splits, shuffle=shuffle_state, random_state=random_num)
    data_dict = dict()
    index = 1
    for train_index, test_index in kf.split(samples, labels):
        x_train, x_test = samples[train_index], samples[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        data_dict[index] = (x_train, x_test, y_train, y_test)
        index += 1
    return data_dict
###################################################################################################

'''
the main method provides a quick demo on the two methods
'''
###################################################################################################
def main():
    # load data
    data_dict = build_data_dict()
    images, labels = build_data_arrays(data_dict)

    # stratified_kfold demo
    print('###################################################################################################')
    data = stratified_kfold(images, labels, num_of_splits=NUM_OF_SPLITS)
    for index, data_tuple in data.items():
        print('index:', index)
        print('y_train', data_tuple[2])
        print('y_test', data_tuple[3])
        print()
    print('###################################################################################################')
    print()

    # kfold demo
    print('###################################################################################################')
    data = kfold(images, labels, num_of_splits=NUM_OF_SPLITS, random_num=42)
    for index, data_tuple in data.items():
        print('index:', index)
        print('y_train', data_tuple[2])
        print('y_test', data_tuple[3])
        print()
    print('###################################################################################################')
###################################################################################################
if __name__ == "__main__":
    main()
