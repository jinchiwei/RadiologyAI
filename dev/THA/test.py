# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import matplotlib
matplotlib.use('Agg') # Added for display errors
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, models
import torch.optim.lr_scheduler as sch
import torch.nn.functional as F

import os, sys
from skimage import io

from PIL import Image
import time

from models import ResNet18_pretrained, inception_v3_pretrained, AlexNet_pretrained, SqueezeNet_pretrained, VGGNet_pretrained, DenseNet_pretrained
from dataset import read_dataset
from test_statistics import roc_auc_metrics
import argparse
from sklearn import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network',
        choices=['resnet18', 'inception_v3', 'alexnet', 'squeezenet', 'vggnet', 'densenet'], default='resnet18',
        help='Choose which neural network to use')
    args = parser.parse_args()

    result_classes = {
        0: 'no_emphysema',
        1: 'yes_emphysema'
    }

    ############ testing ############
    use_gpu = torch.cuda.is_available()
    n_classes = len(result_classes)
    if args.network == 'resnet18':
        model = ResNet18_pretrained(n_classes, freeze=False)
        weightslist = os.listdir('weights/resnet18_weights')
        weightsnum = len(weightslist) - 1
        for weightfile in range(weightsnum):
            load_file = 'weights/resnet18_weights/' + weightslist[weightfile]
            val_data_transform = transforms.Compose([
              transforms.ToPILImage(),
              transforms.Resize((256, 256)),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              # transforms.Normalize(mean=[0.485, 0.456, 0.406],
              #                      std=[0.229, 0.224, 0.225]),
              # transforms.Normalize(mean=[0.4059296, 0.40955055, 0.412535],
              #                      std=[0.21329397, 0.215493, 0.21677108]),
            ])
            test(use_gpu, n_classes, load_file, val_data_transform, model, weightfile)
    elif args.network == 'inception_v3':
        model = inception_v3_pretrained(n_classes, freeze=False)
        weightslist = os.listdir('weights/inception_v3_weights')
        weightsnum = len(weightslist) - 1
        for weightfile in range(weightsnum):
            load_file = 'weights/inception_v3_weights' + weightslist[weightfile]
            val_data_transform = transforms.Compose([
              transforms.ToPILImage(),
              transforms.Resize((300, 300)),
              transforms.CenterCrop(299),
              transforms.ToTensor(),
              # transforms.Normalize(mean=[0.485, 0.456, 0.406],
              #                      std=[0.229, 0.224, 0.225]),
              # transforms.Normalize(mean=[0.4059296, 0.40955055, 0.412535],
              #                      std=[0.21329397, 0.215493, 0.21677108]),
            ])
            test(use_gpu, n_classes, load_file, val_data_transform, model, weightfile)
    elif args.network == 'alexnet':
        model = AlexNet_pretrained(n_classes, freeze=False)
        weightslist = os.listdir('weights/alexnet_weights')
        weightsnum = len(weightslist) - 1
        for weightfile in range(weightsnum):
            load_file = 'weights/alexnet_weights/' + weightslist[weightfile]
            val_data_transform = transforms.Compose([
              transforms.ToPILImage(),
              transforms.Resize((256, 256)),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              # transforms.Normalize(mean=[0.485, 0.456, 0.406],
              #                      std=[0.229, 0.224, 0.225]),
              # transforms.Normalize(mean=[0.4059296, 0.40955055, 0.412535],
              #                      std=[0.21329397, 0.215493, 0.21677108]),
            ])
            test(use_gpu, n_classes, load_file, val_data_transform, model, weightfile)
    elif args.network == 'squeezenet':
        model = SqueezeNet_pretrained(n_classes, freeze=False)
        weightslist = os.listdir('weights/squeezenet_weights')
        weightsnum = len(weightslist) - 1
        for weightfile in range(weightsnum):
            load_file = 'weights/squeezenet_weights/' + weightslist[weightfile]
            val_data_transform = transforms.Compose([
              transforms.ToPILImage(),
              transforms.Resize((256, 256)),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              # transforms.Normalize(mean=[0.485, 0.456, 0.406],
              #                      std=[0.229, 0.224, 0.225]),
              # transforms.Normalize(mean=[0.4059296, 0.40955055, 0.412535],
              #                      std=[0.21329397, 0.215493, 0.21677108]),
            ])
            test(use_gpu, n_classes, load_file, val_data_transform, model, weightfile)
    elif args.network == 'vggnet':
        model = VGGNet_pretrained(n_classes, freeze=False)
        weightslist = os.listdir('weights/vggnet_weights')
        weightsnum = len(weightslist) - 1
        for weightfile in range(weightsnum):
            load_file = 'weights/vggnet_weights/' + weightslist[weightfile]
            val_data_transform = transforms.Compose([
              transforms.ToPILImage(),
              transforms.Resize((256, 256)),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              # transforms.Normalize(mean=[0.485, 0.456, 0.406],
              #                      std=[0.229, 0.224, 0.225]),
              # transforms.Normalize(mean=[0.4059296, 0.40955055, 0.412535],
              #                      std=[0.21329397, 0.215493, 0.21677108]),
            ])
            test(use_gpu, n_classes, load_file, val_data_transform, model, weightfile)
    elif args.network == 'densenet':
        model = DenseNet_pretrained(n_classes, freeze=False)
        weightslist = os.listdir('weights/densenet_weights')
        weightsnum = len(weightslist) - 1
        for weightfile in range(weightsnum):
            load_file = 'weights/densenet_weights/' + weightslist[weightfile]
            val_data_transform = transforms.Compose([
              transforms.ToPILImage(),
              transforms.Resize((256, 256)),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              # transforms.Normalize(mean=[0.485, 0.456, 0.406],
              #                      std=[0.229, 0.224, 0.225]),
              # transforms.Normalize(mean=[0.4059296, 0.40955055, 0.412535],
              #                      std=[0.21329397, 0.215493, 0.21677108]),
            ])
            test(use_gpu, n_classes, load_file, val_data_transform, model, weightfile)


def test(use_gpu, n_classes, load_file, val_data_transform, model, weightfile):
    batch_size=10
    model.load_state_dict(torch.load(os.path.join('./', load_file)))
    radio_val = read_dataset(mode='test', transform=val_data_transform)
    radio_data_loader = DataLoader(radio_val, batch_size=batch_size, shuffle=True, num_workers=2)

    model.train(False)

    running_corrects = 0
    total = len(radio_val.sample_paths)
    print(total)

    def do_gpu(x):
        return x.cuda() if use_gpu else x

    if use_gpu:
        model = model.cuda()

    TP = 0  # pred true, label true
    TN = 0  # pred false, label false
    FP = 0  # pred true, label false
    FN = 0  # pred false, label true

    y_true = []
    y_score = []
    for data in radio_data_loader:
        inputs, labels = data

        """
        # show first images of the batch
        plt.imshow(np.transpose(inputs.numpy()[0], (1,2,0)))
        plt.show()
        """

        original = inputs
        inputs = Variable(do_gpu(inputs)).float()
        labels = Variable(do_gpu(labels)).long()

        # forward
        outputs = model(inputs)

        local_y_score = F.softmax(outputs, 1)

        if use_gpu:
            y_score.append(local_y_score.data.cpu().numpy())
            y_true.append(labels.data.cpu().numpy())
        else:
            y_score.append(local_y_score.data.numpy())
            y_true.append(labels.data.numpy())

        _, preds = torch.max(outputs.data, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)

        # ROC curve analysis
        preds = preds.float().cpu().numpy()
        labels = labels.data.float().cpu().numpy()
        TP += np.sum(np.logical_and(preds == 1.0, labels == 1.0))
        TN += np.sum(np.logical_and(preds == 0.0, labels == 0.0))
        FP += np.sum(np.logical_and(preds == 1.0, labels == 0.0))
        FN += np.sum(np.logical_and(preds == 0.0, labels == 1.0))

        """
        # show incorrectly classified images
        for idx in range(len(original)):
          if (preds != labels.data)[idx]:
            plt.imshow(np.transpose(original.numpy()[idx], (1,2,0)))
            plt.show()
            # print('here', idx)
        """

    print('---------  correct: {:03d} -----------'.format(running_corrects))
    print('---------  total: {:03d} -----------'.format(total))
    print('---------  accuracy: {:.4f} -----------'.format(float(running_corrects)/total))

    output = open('test_result' + str(weightfile) + '.txt', 'w')

    output.write('---------  correct: {:03d} -----------'.format(running_corrects) + "\n")
    output.write('---------  total: {:03d} -----------'.format(total) + "\n")
    output.write('---------  accuracy: {:.4f} -----------'.format(float(running_corrects) / total) + "\n")

    if n_classes < 3:  # roc/auc for binary output
        roc_auc_metrics(y_true, y_score, n_classes, weightfile)  # call statistics file for roc/auc

    sensitivity  = TP / (TP + FN)
    specificity  = TN / (TN + FP)
    pos_like_ratio = sensitivity / (1 - specificity)
    neg_like_ratio = (1 - sensitivity) / specificity
    pos_pred_val = TP / (TP + FP)
    neg_pred_val = TN / (TN + FN)

    print('sensitivity: %f\nspecificity:'
          '%f\npositive likelihood value: %f\nnegative likelihood value:'
          '%f\npositive predictive value: %f\nnegative predictive value:'
          '%f\nTP: %f\nTN: %f\nFP: %f\nFN: %f'
          % (sensitivity, specificity, pos_like_ratio, neg_like_ratio, pos_pred_val, neg_pred_val, TP, TN, FP, FN))

    output.write(
        'sensitivity: %f\nspecificity:'
        '%f\npositive likelihood value: %f\nnegative likelihood value:'
        '%f\npositive predictive value: %f\nnegative predictive value:'
        '%f\nTP: %f\nTN: %f\nFP: %f\nFN: %f'
        % (sensitivity, specificity, pos_like_ratio, neg_like_ratio, pos_pred_val, neg_pred_val, TP, TN, FP, FN))
    output.close()


if __name__ == '__main__':
    main()


