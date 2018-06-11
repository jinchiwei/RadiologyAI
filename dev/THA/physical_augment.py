import os, sys
from skimage import io, color, transform
import numpy as np
# import cv2
from matplotlib import pyplot as plt


result_classes = {
    0: 'no_THA',
    1: 'yes_THA',
    2: 'yes_HRA'
}

dataset_dir = 'dataset/100_20_30'
directories = {}
for class_num in result_classes:
    directories['train_' + str(class_num)] = result_classes[class_num] + '_train'
    directories['val_' + str(class_num)] = result_classes[class_num] + '_val'
    # directories['test_' + str(class_num)] = result_classes[class_num] + '_test'

# dataset_dir = 'dataset/100_20_30_aug'
# directories = {'no_train' : 'no_THA_train',
#                 'yes_train' : 'yes_THA_train',
#                 'no_val' : 'no_THA_val',
#                 'yes_val' : 'yes_THA_val'
#               }
#                 # 'no_test' : 'no_THA_test',
#                 # 'yes_test' : 'yes_THA_test'


def read_image(img_path):
    x = io.imread(img_path)
    img_name = os.path.splitext(img_path)[0]
    extension = os.path.splitext(img_path)[1]
    return x, img_name, extension


def save_image(x, new_img_path, extension):
    # plt.imshow(x)
    # plt.show()
    # print(new_img_path)
    # print(extension)
    if extension == '':
        extension = '.jpg'
    new_img_path = new_img_path + extension
    print(new_img_path)
    if x.dtype == np.uint16:
        x = x.astype(np.uint8)
    io.imsave(new_img_path, x)


def rotate_image(img_path, degree):
    x, img_name, extension = read_image(img_path)
    x = transform.rotate(x, degree, resize=True)
    new_img_path = img_name + '_rotate_' + str(degree)
    save_image(x, new_img_path, extension)


def affine_image(img_path, iteration, rotate=False):
    x, img_name, extension = read_image(img_path)

    # original configuration from
    # https://www.programcreek.com/python/example/96400/skimage.transform.AffineTransform

    # rotation = np.random.random_integers(0, 360)
    # translation = (np.random.random_integers(-20, 20), np.random.random_integers(-20, 20))
    # scale = (np.random.uniform(1/1.2, 1.2), np.random.uniform(1/1.2, 1.2))
    # shear = np.random.random_integers(-10, 10)

    # tf_augment = transform.AffineTransform(scale=scale, rotation=np.deg2rad(rotation), translation=translation, shear=np.deg2rad(shear))
    # x = transform.warp(x, tf_augment, order=1, preserve_range=True, mode='symmetric')

    translation = (np.random.random_integers(-20, 20), np.random.random_integers(-20, 20))
    scale = (np.random.uniform(1/1.2, 1.2), np.random.uniform(1/1.2, 1.2))
    shear = np.random.random_integers(-10, 10)

    rotation = 0
    if rotate:
        rotation = np.random.random_integers(-15, +15)

    tf_augment = transform.AffineTransform(scale=scale, rotation=np.deg2rad(rotation), translation=translation, shear=np.deg2rad(shear))
    x = transform.warp(x, tf_augment)
    new_img_path = img_name + '_affine_' + str(iteration) + '_' + str(rotate)

    save_image(x, new_img_path, extension)


def amplify_image(img_path):
    for angle in range(30, 331, 30):
        rotate_image(img_path, angle)
    for iteration in range(1, 2):
        affine_image(img_path, iteration, rotate=False)
        affine_image(img_path, iteration, rotate=True)


def process_images():
    for key in directories:
        img_folder = os.path.join(dataset_dir, directories[key])
        images = os.listdir(img_folder)
        for image in images:
            if not image.startswith('.'): # avoid .DS_Store
                img_path = os.path.join(img_folder, image)
                amplify_image(img_path)


if __name__ == '__main__':
    process_images()
    # for key in directories:
    #   img_folder = os.path.join(dataset_dir, directories[key])
    #   images = os.listdir(img_folder)
    #   for image in images:
    #     if not image.startswith('.'): # avoid .DS_Store
    #       img_path = os.path.join(img_folder, image)
    #       amplify_image(img_path)
    #       break;
    #   break;
