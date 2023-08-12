import os
import time
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
from data_loader.ImageNet_datasets import ImageNetData
from data_loader.driver_dataset import DriverDataset
import model.resnet_cbam4 as resnet_cbam
from trainer.trainer import Trainer
from utils.logger import Logger
from PIL import Image
from torchnet.meter import ClassErrorMeter
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from utils.utils import validate, show_confMat
import d2lzh_pytorch as d2l
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[96, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[96, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=(2, 2)),      # output[256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[256, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=(1, 1)),     # output[384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=(1, 1)),     # output[384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=(1, 1)),     # output[256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[256, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()


if __name__ == '__main__':

    # 数据增强
    # transforms.RandomHorizontalFlip(),
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    data_root = '/home/cwh/labs/my_cbam/model/'
    train_datasets = DriverDataset(data_root, "Train_data_list.csv", transform=data_transforms['train'])
    val_datasets = DriverDataset(data_root, "Test_data_list.csv", transform=data_transforms['val'])
    print("train: {}, test: {}".format(len(train_datasets), len(val_datasets)))
    batch_size = 64
    train_dataloaders = data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloaders = data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False, num_workers=8)
    print("train dataloader: {}, val dataloader: {}".format(len(train_dataloaders), len(val_dataloaders)))

    num_classes = 5

    lr, num_epochs = 0.001, 1
    model = AlexNet(num_classes)
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d2l.train_ch5(net, train_dataloaders, val_dataloaders, batch_size, optimizer, device, num_epochs)

    print(sum([param.nelement() for param in net.parameters()]))



