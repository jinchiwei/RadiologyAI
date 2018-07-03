# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2

import torch
import os, sys
from models import ResNet18_pretrained, ResNet50_pretrained


# input images
############ dataloader ############
result_classes = {
    0: 'no_THA',
    1: 'yes_THA',
    # 2: 'yes_HRA'
}

# image_file = os.listdir('cam')[0]

# dataset_dir = 'dataset/100_20_30'
# directories = {}
# for class_num in result_classes:
#     directories['train_' + str(class_num)] = result_classes[class_num] + '_train'
#     directories['val_' + str(class_num)] = result_classes[class_num] + '_val'
#     directories['test_' + str(class_num)] = result_classes[class_num] + '_test'

###  output directories
output_dir = 'cam'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

###  choose model
model_id = 3 # we are using ResNet50 by default
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features'
elif model_id == 2:
    net = ResNet18_pretrained(2, freeze=False)
    weightslist = os.listdir('weights/resnet18_weights')
    weightsnum = len(weightslist)
    for weightfile in range(weightsnum):
        if not weightslist[weightfile].startswith('LOG'):  # avoid LOG.txt
            load_file = 'weights/resnet18_weights/' + weightslist[weightfile]
    net.load_state_dict(torch.load(os.path.join('./', load_file)))
    finalconv_name = 'layer4'
elif model_id == 3:
    net = ResNet50_pretrained(2, freeze=False)
    weightslist = os.listdir('weights/resnet50_weights')
    weightsnum = len(weightslist)
    for weightfile in range(weightsnum):
        if not weightslist[weightfile].startswith('LOG'):  # avoid LOG.txt
            load_file = 'weights/resnet50_weights/' + weightslist[weightfile]
    net.load_state_dict(torch.load(os.path.join('./', load_file)))
    finalconv_name = 'layer4'
net.eval()

### preprocessing
preprocess = transforms.Compose([
  transforms.Resize((256, 256)),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
])


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


# ################ SINGLE IMAGE
#
# #image_url = os.path.join(dataset_dir, directories['train_1'])
# #samples = os.listdir(image_url)
#
# # hook the feature extractor
# features_blobs = []
# def hook_feature(module, input, output):
#     features_blobs.append(output.data.cpu().numpy())
#
# net._modules.get(finalconv_name).register_forward_hook(hook_feature)
#
# # get the softmax weight
# params = list(net.parameters())
# weight_softmax = np.squeeze(params[-2].data.numpy())
#
# img_name = os.path.splitext(image_file)[0]
#
# #image_file = os.path.join(image_url, sample)
# image_url = os.path.join('cam/', image_file)
# img_pil = Image.open(image_url).convert('RGB')
# img_pil.save(output_dir + '/' + img_name + '.jpg')
#
# img_tensor = preprocess(img_pil)
# img_variable = Variable(img_tensor.unsqueeze(0))
# logit = net(img_variable)
#
# # imagenet category list
# classes = {int(key):value for (key, value) in result_classes.items()}
#
# h_x = F.softmax(logit).data.squeeze()
# probs, idx = h_x.sort(0, True)
#
# # output the prediction
# #for i in range(0, 2):
# #    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
#
# # generate class activation mapping for the top1 prediction
# CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
#
# # render the CAM and output
# #print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
# img = cv2.imread(output_dir + '/' + img_name + '.jpg')
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.3 + img * 0.5
# cv2.imwrite(output_dir + '/' + img_name + '_CAM' + '.jpg', result)
# ################ SINGLE IMAGE


################ REPETITION
# image_url = os.path.join(dataset_dir, directories['train_1'])
image_url = 'cam'
samples = os.listdir(image_url)
for sample in samples:

    if sample.startswith('.'): # avoid .DS_Store
        continue

    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    img_name = os.path.splitext(sample)[0]

    image_file = os.path.join(image_url, sample)
    img_pil = Image.open(image_file).convert('RGB')
    img_pil.save(output_dir + '/' + img_name + '.jpg')

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    # imagenet category list
    classes = {int(key):value for (key, value) in result_classes.items()}

    h_x = F.softmax(logit).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # # output the prediction
    # for i in range(0, 2):
    #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread(output_dir + '/' + img_name + '.jpg')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(output_dir + '/' + img_name + '_CAM' + '.jpg', result)
################ REPETITION






