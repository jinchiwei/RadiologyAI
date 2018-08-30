import torch.nn as nn
from torchvision import datasets, transforms, models

# from googlenet import GoogLeNet


def ResNet18_pretrained(n_classes, freeze=True):
    model = models.__dict__['resnet18'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, n_classes)
    return model



def ResNet50_pretrained(n_classes, freeze=True):
    model = models.resnet50(pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    num_filters = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_filters, n_classes * 3),
        nn.Linear(n_classes * 3, n_classes * 2),
        nn.Linear(n_classes * 2, n_classes)
    )
    return model

def ResNet152_pretrained(n_classes, freeze=True):
    model = models.resnet152(pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, n_classes)
    return model

def inception_v3_pretrained(n_classes, freeze=True):
    model = models.__dict__['inception_v3'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, n_classes)
    return model


def AlexNet_pretrained(n_classes, freeze=True):
    model = models.__dict__['alexnet'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    lastclasslayer = str(len(model.classifier._modules) - 1)
    num_filters = model.classifier._modules[lastclasslayer].in_features
    model.classifier._modules[lastclasslayer] = nn.Linear(num_filters, n_classes)
    return model


def SqueezeNet_pretrained(n_classes, freeze=True):
    model = models.__dict__['squeezenet1_1'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    model.classifier._modules['1'] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes
    return model


def VGGNet_pretrained(n_classes, freeze=True):
    model = models.__dict__['vgg16'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    lastclasslayer = str(len(model.classifier._modules) - 1)
    num_filters = model.classifier._modules[lastclasslayer].in_features
    model.classifier._modules[lastclasslayer] = nn.Linear(num_filters, n_classes)
    return model


def DenseNet_pretrained(n_classes, freeze=True):
    model = models.__dict__['densenet161'](pretrained=True)
    # if weights are not frozen, train
    for param in model.parameters():
        if freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # change last layer to output n_classes
    num_filters = model.classifier.in_features
    model.classifier = nn.Linear(num_filters, n_classes)
    return model

class FC_Layer_Binary(nn.Module):
    def __init__(self, input_size):
        super(FC_Layer_Binary, self).__init__()
        self.fc = nn.Linear(input_size, 2)
    def forward(self, x):
        x = self.fc(x)
        return x

class FC_Layer_14(nn.Module):
    def __init__(self, input_size):
        super(FC_Layer_14, self).__init__()
        self.fc = nn.Linear(input_size, 14)
    def forward(self, x):
        x = self.fc(x)
        return x
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

  
