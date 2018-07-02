import torch
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
import argparse
import os

# add command line arguments on what folder should be checked
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Choose which img folder to use', required=True)
parser.add_argument('-m', '--model', help='Choose model weight file', required=True)
args = parser.parse_args()

# add the path to the directory
path = "./" + args.folder + "/"
result_classes = {
    0: 'no_THA',
    1: 'yes_THA',
}
modelPath = "./" + args.model
softmax3 = nn.Softmax(dim=3)
softmax1 = nn.Softmax(dim=1)

def main():
    data_transform = transforms.Compose([                
                  transforms.Resize((256, 256)),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                ])
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 2)
    model.load_state_dict(torch.load(modelPath, map_location=lambda storage, loc: storage))
    model.eval()
    for filename in os.listdir(path):
        img = Image.open(path + filename).convert('RGB')
        img_tensor = data_transform(img)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        fc_out = model(img_variable)
        print(filename + ": " + result_classes[fc_out.data.numpy().argmax()])

if __name__ == '__main__':
    main()
