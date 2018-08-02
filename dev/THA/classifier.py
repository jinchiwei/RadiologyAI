import torch
from torchvision import datasets, transforms  # , models
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
import argparse
import os
from shutil import copyfile
from models import ResNet18_pretrained, ResNet50_pretrained


# add command line arguments on what folder should be checked
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', default='classifier', help='Choose which img folder to use')  # required=True
parser.add_argument('-m', '--model', default='resnet50', help='Choose model weight file')  # required=True
args = parser.parse_args()

# add the path to the directory
path = "./" + args.folder + "/"
result_classes = {
    0: 'no_THA',
    1: 'yes_THA',
}
n_classes = len(result_classes)

use_gpu = torch.cuda.is_available()
modelPath = "./" + args.model
softmax3 = nn.Softmax(dim=3)
softmax1 = nn.Softmax(dim=1)

for class_num in result_classes:
    output_dir = result_classes[class_num]
    if not os.path.exists(path + '/' + output_dir):
        os.makedirs(path + '/' + output_dir)
        # os.chmod(path + '/' + output_dir, 0o777)


def main():
    data_transform = transforms.Compose([                
                  transforms.Resize((256, 256)),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                ])
    # model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(2048, 2)
    model = ResNet50_pretrained(n_classes, freeze=False)
    # model.load_state_dict(torch.load(modelPath, map_location=lambda storage, loc: storage))
    weightslist = os.listdir('weights/resnet50_weights')
    weightsnum = len(weightslist) - 1
    if weightslist[weightsnum].startswith('LOG'):  # avoid LOG.txt
        weightsnum = weightsnum - 1
    load_file = 'weights/resnet50_weights/' + weightslist[weightsnum]
    # load_file = 'weights/resnet50_weights/010_1.000.pkl'
    # for weightfile in range(weightsnum):
        # if not weightslist[weightfile].startswith('LOG'):  # avoid LOG.txt
            # load_file = 'weights/resnet50_weights/' + weightslist[weightfile]

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using " + str(torch.cuda.device_count()) + ' GPU(s)')
        if torch.cuda.device_count() > 1:
            gpu_ids = list(range(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=gpu_ids).cuda()
        # else:
        #     model = model.cuda()

    model.load_state_dict(torch.load(os.path.join('./', load_file)))
    model.eval()

    for filename in os.listdir(path):
        if os.path.isdir(path + filename):  # skip directories
            continue
        img = Image.open(path + filename).convert('RGB')
        img_tensor = data_transform(img)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        fc_out = model(img_variable)

        logit = model(img_variable)
#        # imagenet category list
#        # classes = {int(key): value for (key, value) in result_classes.items()}
#
#        h_x = F.softmax(logit).data.squeeze()
#        probs, idx = h_x.sort(0, True)
#        print(probs)
#
#        # # output the prediction
#        # for i in range(0, 2):
#        #    print('{:.3f} -> {}'.format(probs[i], classes[idx[i].item()]))

        # if use_gpu:
        if torch.cuda.device_count() > 1:
            prediction = fc_out.data.cpu().numpy().argmax()
        else:
            prediction = fc_out.data.numpy().argmax()
        print(filename + ": " + ' -> ' + str(nn.functional.softmax(fc_out.data)) + ' -> ' + result_classes[prediction])
        copyfile(path + "/" + filename, path + "/" + result_classes[prediction] + "/" + filename)


if __name__ == '__main__':
    main()
