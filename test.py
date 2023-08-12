import os

from PIL import Image
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, models
from model import *

from data_loader.driver_dataset import DriverDataset

import model.resnet_fdan as resnet_fdan
import model.resnet_adnet as resnet_adnet

data_root = '/home/cwh/data/xwc博士代码2/data_HNU/'
RESULT_FILE = 'result.csv'

def load_state_dict(model_dir, is_multi_gpu):
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]       # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def test_and_generate_result(model_name='resnet18_fdan', num_classes = 10, img_size=224):

    data_transform = {
        'test': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    is_use_cuda = torch.cuda.is_available()

    if  'resnet18_fdan' ==model_name:
        my_model = resnet_fdan.resnet18_fdan(pretrained=True, num_classes=num_classes)
    elif 'alexnet' == model_name:
        my_model = models.alexnet(pretrained=False, num_classes=num_classes)
    elif 'resnet18_adnet' == model_name:
        my_model = resnet_adnet.resnet18_adnet(pretrained=True, num_classes=num_classes)
    elif 'resnet18_se' == model_name:
        my_model = resnet_se.resnet18_se(pretrained=True, num_classes=num_classes)
    elif 'resnet18' == model_name:
        my_model = resnet_se.resnet18_se(pretrained=True, num_classes=num_classes)
    elif 'fcnn18' == model_name:
        my_model = resnet_se.resnet18_se(pretrained=True, num_classes=num_classes)
    elif 'resnet18_GMM' == model_name:
        my_model = resnet_GMM.resnet18(pretrained=True, num_classes=num_classes)
    elif 'tscnn' == model_name:
        my_model = resnet_tscnn.resnet18(pretrained=True, num_classes=num_classes)      
    else:
        raise ModuleNotFoundError

    if is_use_cuda: 
        my_model = my_model.cuda()

    state_dict = torch.load('./checkpoint/' + model_name + '.ckpt', map_location=lambda storage, loc: storage.cuda())['state_dict']
    my_model.load_state_dict(state_dict)

    my_model.eval()

    with open(os.path.join('checkpoint', model_name, model_name+'_'+str(img_size)+'_'+RESULT_FILE), 'w', encoding='utf-8') as fd:
        fd.write('filename|defect,probability\n')
        test_files_list = DriverDataset(data_root, "Test_data_list_N.csv", transform=data_transform['test'])
        for _file in test_files_list:
            file_name = _file
            if '.jpg' not in file_name:
                continue
            file_path = os.path.join(DATA_ROOT, file_name)
            img_tensor = data_transform(Image.open(file_path).convert('RGB')).unsqueeze(0)
            
            if is_use_cuda:
                img_tensor = Variable(img_tensor.cuda(), volatile=True)
            _, output, _ = my_model(img_tensor)
            
            output = F.softmax(output, dim=1)
            for k in range(11):
                defect_prob = round(output.data[0, k], 6)
                if defect_prob == 0.:
                    defect_prob = 0.000001
                elif defect_prob == 1.:
                    defect_prob = 0.999999
                target_str = '%s,%.6f\n' % (file_name + '|' + ('norm' if 0 == k else 'defect_'+str(k)), defect_prob)
                fd.write(target_str)

if __name__ == '__main__':   
    model_name = 'resnet18_fdan'
    num_classes = 10
    img_size=224
    test_and_generate_result(model_name,num_classes, img_size)
   
