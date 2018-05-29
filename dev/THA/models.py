import torch.nn as nn
from torchvision import datasets, transforms, models

# from googlenet import GoogLeNet


def ResNet18_pretrained(n_classes, freeze=True):
  model = models.__dict__['resnet18'](pretrained=True)
  ## freeze all weights
  if freeze:
    for param in model.parameters():
      param.requires_grad = False
  else:
    for param in model.parameters():
      param.requires_grad = True

  ## change the last 1000-fc to n_classes
  num_filters = model.fc.in_features
  model.fc = nn.Linear(num_filters,n_classes)
  return model



def inception_v3_pretrained(n_classes, freeze=True):
  model = models.__dict__['inception_v3'](pretrained=True)
  if freeze:
    for param in model.parameters():
      param.requires_grad = False
  else:
    for param in model.parameters():
      param.requires_grad = True

  ## change the last 1000-fc to n_classes
  num_filters = model.fc.in_features
  model.fc = nn.Linear(num_filters,n_classes)
  return model



def AlexNet_pretrained(n_classes, freeze=True):
  model = models.__dict__['alexnet'](pretrained=True)
  if freeze:
    for param in model.parameters():
      param.requires_grad = False
  else:
    for param in model.parameters():
      param.requires_grad = True

  ## change the last 1000-fc to n_classes
  num_filters = model.fc.in_features
  model.fc = nn.Linear(num_filters, n_classes)
  return model



def SqueezeNet_pretrained(n_classes, freeze=True):
  model = models.__dict__['squeezenet1_1'](pretrained=True)
  if freeze:
    for param in model.parameters():
      param.requires_grad = False
  else:
    for param in model.parameters():
      param.requires_grad = True

  ## change the last 1000-fc to n_classes
  num_filters = model.fc.in_features
  model.fc = nn.Linear(num_filters, n_classes)
  return model



def VGGNet_pretrained(n_classes, freeze=True):
  model = models.__dict__['vgg16'](pretrained=True)
  if freeze:
    for param in model.parameters():
      param.requires_grad = False
  else:
    for param in model.parameters():
      param.requires_grad = True

  ## change the last 1000-fc to n_classes
  num_filters = model.fc.in_features
  model.fc = nn.Linear(num_filters, n_classes)
  return model



def DenseNet_pretrained(n_classes, freeze=True):
  model = models.__dict__['densenet161'](pretrained=True)
  if freeze:
    for param in model.parameters():
      param.requires_grad = False
  else:
    for param in model.parameters():
      param.requires_grad = True

  ## change the last 1000-fc to n_classes
  num_filters = model.fc.in_features
  model.fc = nn.Linear(num_filters, n_classes)
  return model



# def GoogLeNet_pretrained(n_classes, freeze=True):
#   model = GoogLeNet();
#   if freeze:
#     for param in model.parameters():
#       param.requires_grad = False
#   else:
#     for param in model.parameters():
#       param.requires_grad = True

#   model.linear = nn.Linear(1024, n_classes)
#   return model

if __name__ == '__main__':
  # model = ResNet18_pretrained(2)
  model = inception_v3_pretrained(2)
  print("success until here")

  
