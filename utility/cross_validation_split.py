import os
import shutil
from dataset_utils import CrossValidationUtils

# data directories
src_data_dir = "data/source"
cv_split_data_dir = 'data/cross_validation_110718'

# split parameters
K = 5
TRAIN = 0.7
TEST = 0.2
SEED = 0

# 1. build the required directories for the cross validation
########################################################################################
if not os.path.exists(cv_split_data_dir):
    os.mkdir(cv_split_data_dir)
for split_index in range(K):
    current_split_dir_name = "split_{}_data".format(split_index+1)
    current_split_dir = os.path.join(cv_split_data_dir, current_split_dir_name)
    if not os.path.exists(current_split_dir):
        os.mkdir(current_split_dir)

    current_split_train_dir = os.path.join(current_split_dir, "train")
    if not os.path.exists(current_split_train_dir):
        os.mkdir(current_split_train_dir)
    current_split_validation_dir = os.path.join(current_split_dir, 'validation')
    if not os.path.exists(current_split_validation_dir):
        os.mkdir(current_split_validation_dir)
    current_split_test_dir = os.path.join(current_split_dir, 'test')
    if not os.path.exists(current_split_test_dir):
        os.mkdir(current_split_test_dir)

    current_split_train_od_dir = os.path.join(current_split_train_dir, 'OD')
    if not os.path.exists(current_split_train_od_dir):
        os.mkdir(current_split_train_od_dir)
    current_split_train_os_dir = os.path.join(current_split_train_dir, 'OS')
    if not os.path.exists(current_split_train_os_dir):
        os.mkdir(current_split_train_os_dir)
    current_split_validation_od_dir = os.path.join(current_split_validation_dir, 'OD')
    if not os.path.exists(current_split_validation_od_dir):
        os.mkdir(current_split_validation_od_dir)
    current_split_validation_os_dir = os.path.join(current_split_validation_dir, 'OS')
    if not os.path.exists(current_split_validation_os_dir):
        os.mkdir(current_split_validation_os_dir)
    current_split_test_od_dir = os.path.join(current_split_test_dir, 'OD')
    if not os.path.exists(current_split_test_od_dir):
        os.mkdir(current_split_test_od_dir)
    current_split_test_os_dir = os.path.join(current_split_test_dir, 'OS')
    if not os.path.exists(current_split_test_os_dir):
        os.mkdir(current_split_test_os_dir)
########################################################################################

# 2. build data_dict
data_dict = dict()
od_image_dir = os.path.join(src_data_dir, "All OD for 5 fold cross validation (9-11-18)")   # 291
os_image_dir = os.path.join(src_data_dir, "All OS for 5 fold cross validation (9-11-18)")   # 285
for image in os.listdir(od_image_dir):
    image_path = os.path.join(od_image_dir, image)
    data_dict[image_path] = 0
for image in os.listdir(os_image_dir):
    image_path = os.path.join(os_image_dir, image)
    data_dict[image_path] = 1

# 3. build the split_records dictionary
split_records = CrossValidationUtils.stratified_kfold_cv(
                        data=data_dict,
                        train_ratio=TRAIN, test_ratio=TEST,
                        num_splits=K, random_num=SEED)

# 4. organize the source images in the proper directories
########################################################################################
for split_index, split_record in split_records.items():
    current_split_dir_name = 'split_{}_data'.format(split_index)
    current_split_dir = os.path.join(cv_split_data_dir, current_split_dir_name)
    current_train_data = split_record['train']
    current_validation_data = split_record['validation']
    current_test_data = split_record['test']

    current_split_train_dir = os.path.join(current_split_dir, 'train')
    for image, label in zip(current_train_data[0], current_train_data[1]):
        if label == 0:
            dest_path = os.path.join(current_split_train_dir, 'OD')
            shutil.copy(image, dest_path)
        elif label == 1:
            dest_path = os.path.join(current_split_train_dir, 'OS')
            shutil.copy(image, dest_path)
        else:
            print('{} is not properly labeled as OD or OS'.format(image))
            sys.exit(1)
    current_split_validation_dir = os.path.join(current_split_dir, 'validation')
    for image, label in zip(current_validation_data[0], current_validation_data[1]):
        if label == 0:
            dest_path = os.path.join(current_split_validation_dir, 'OD')
            shutil.copy(image, dest_path)
        elif label == 1:
            dest_path = os.path.join(current_split_validation_dir, 'OS')
            shutil.copy(image, dest_path)
        else:
            print('{} is not properly labeled as OD or OS'.format(image))
            sys.exit(1)
    current_split_test_dir = os.path.join(current_split_dir, 'test')
    for image, label in zip(current_test_data[0], current_test_data[1]):
        if label == 0:
            dest_path = os.path.join(current_split_test_dir, 'OD')
            shutil.copy(image, dest_path)
        elif label == 1:
            dest_path = os.path.join(current_split_test_dir, 'OS')
            shutil.copy(image, dest_path)
        else:
            print('{} is not properly labeled as OD or OS'.format(image))
            sys.exit(1)

########################################################################################
