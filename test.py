import os
from collections import OrderedDict
from PIL import Image
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, models
from model import *
# import pretrainedmodels
import numpy as np
import model.resnet_fdan as resnet_fdan
import model.resnet_adnet as resnet_adnet
import model.resnet_se as resnet_se
import model.resnet as resnet
import model.fcnn as fcnn
import csv

data_root = '/home/cwh/data/datasets/data_HNU/'
RESULT_FILE = 'result.csv'

import warnings
warnings.filterwarnings("ignore")


def test_and_generate_result_round(model_name='resnet18_fdan', num_classes=5, img_size=224):
    data_transform = transforms.Compose([
        transforms.Resize((img_size,img_size),Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    is_use_cuda = torch.cuda.is_available()

    if 'resnet18_fdan' == model_name:
        my_model = resnet_fdan.resnet18_fdan(pretrained=True, num_classes=num_classes)
    elif 'alexnet' == model_name:
        my_model = models.alexnet(pretrained=False, num_classes=num_classes)
    elif 'resnet18_adnet' == model_name:
        my_model = resnet_adnet.resnet18_adnet(pretrained=True, num_classes=num_classes)
    elif 'resnet18_se' == model_name:
        my_model = resnet_se.resnet18_se(pretrained=True, num_classes=num_classes)
    elif 'fcnn18' == model_name:
        my_model = fcnn.fcnn18(pretrained=True, num_classes=num_classes)
    elif 'resnet18' == model_name:
        my_model = resnet.resnet18(pretrained=True, num_classes=num_classes)
    else:
        raise ModuleNotFoundError
 
    #print('./checkpoint/' + model_name + '/Models_epoch_' + epoch_num + '.ckpt')
    state_dict = torch.load('./checkpoint/' + model_name  + '.ckpt', map_location=lambda storage, loc: storage.cuda())['state_dict']
 
 
    my_model.load_state_dict(state_dict)
 
    if is_use_cuda:
        my_model = my_model.cuda()
    my_model.eval()
 
 
    with open(os.path.join('checkpoint', model_name+'_'+ RESULT_FILE), 'w', encoding='utf-8') as fd:

        fd.write('No. ' + 'image_name ' + 'gt'  + 'pred')

        ## Read test images
        test_file_csv = open(os.path.join(data_root, "Test_data_list.csv"))
        test_files_list = csv.reader(test_file_csv)

        ii = 0  # Number of pictures being processed
        jj = 0  # the number of correctly predicted images
        for test_file in test_files_list:
            #print(test_file[0], test_file[1])
            file_path = os.path.join(data_root, test_file[0])
            #print(file_path, test_file[1])

            img_tensor = data_transform(Image.open(file_path).convert('RGB')).unsqueeze(0)
            # print("5667",img_tensor)

            if is_use_cuda:
                img_tensor = Variable(img_tensor.cuda(), volatile=True)
            # _, output, _ = my_model(img_tensor)
            # print( ":", img_tensor.shape)
            output = my_model(img_tensor)
            # print( "2222222222222:", output.data)
            output = F.softmax(output, dim=1)

            # print( "33333333333333:", output.data[0, 0])
            output = Variable(output)
            output = output.cpu().numpy()
            # print("Probability: ", output)

            pred = np.argmax(output)

            if pred.item() == int(test_file[1]):
                jj += 1

            print(ii, test_file[0], int(test_file[1]), pred)
            fd.write(str(ii) + ' ' + test_file[0]+ ' ' + test_file[1] + ' ' + str(pred) + '\n')

            ii += 1

        acc = jj / ii * 100
        print("Accuracy: %.4f%%" % acc)
        fd.write(str(acc))


if __name__ == '__main__':
    model_name = 'resnet18'  #'resnet18_fdan'
    num_classes = 5
    img_size = 224
    test_and_generate_result_round(model_name, num_classes, img_size)

    
    
    
