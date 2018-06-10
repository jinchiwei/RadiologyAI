import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

class ResNet18(nn.Module):
  def __init__(self, n_classes1, n_classes2, freeze=True):
    super(ResNet18, self).__init__()
    
    self.model = models.__dict__['resnet18'](pretrained=True)
    if freeze:
      for param in self.model.parameters():
        param.requires_grad = False
    else:
      for param in self.model.parameters():
        param.requires_grad = True

    self.conv1 = self.model.conv1
    self.bn1 = self.model.bn1
    self.relu = self.model.relu
    self.maxpool = self.model.maxpool

    self.layer1 = self.model.layer1
    self.layer2 = self.model.layer2
    self.layer3 = self.model.layer3
    self.layer4 = self.model.layer4

    self.avgpool = self.model.avgpool
    num_filters = self.model.fc.in_features

    #####
    self.fc1 = nn.Linear(num_filters, n_classes1) # gender 0, 1
    self.fc2 = nn.Linear(num_filters, n_classes2) # age bucket 0, 1, ..., 19

  def forward(self, x):
    """
    # original ResNet18 forward function
    def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)

      x = self.avgpool(x)
      x = x.view(x.size(0), -1)
      x = self.fc(x)

      return x
    """
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1) # output: 1000
    
    x_gender = self.fc1(x)
    x_age = self.fc2(x)

    return x_gender, x_age



def ResNet18_pretrained(n_classes1, n_classes2, freeze=True):
  model = ResNet18(n_classes1, n_classes2, freeze)
  return model



if __name__ == '__main__':
  model = ResNet18_pretrained(2, 20, freeze=True)
  print("success until here")











