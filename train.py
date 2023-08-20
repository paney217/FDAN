import os
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
from data_loader.ImageNet_datasets import ImageNetData
from data_loader.driver_dataset import DriverDataset
import model.resnet_fdan as resnet_fdan
import model.resnet_adnet as resnet_adnet
import model.resnet as resnet
import model.resnet_se as resnet_se
import model.fcnn as fcnn


from trainer.trainer import Trainer
from utils.logger import Logger
from PIL import Image
from torchnet.meter import ClassErrorMeter
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.utils.data as data


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


def main(args):
    if 0 == len(args.resume):
        logger = Logger('./logs/'+args.model+'.log')
    else:
        logger = Logger('./logs/'+args.model+'.log', True)

    logger.append(vars(args))

    if args.display:
        writer = SummaryWriter()
    else:
        writer = None

    gpus = args.gpu.split(',')
    # 数据增强
    # transforms.RandomHorizontalFlip(),
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    #train_datasets = datasets.ImageFolder(os.path.join(args.data_root, 't256'), data_transforms['train'])
    #val_datasets   = datasets.ImageFolder(os.path.join(args.data_root, 'v256'), data_transforms['val'])
    data_root = '/home/cwh/data/datasets/data_HNU/'  #'/home/cwh/data/xwc博士代码2/data_HNU/' 
    train_datasets = DriverDataset(data_root, "Train_data_list.csv", transform=data_transforms['train'])  #Train_data_list.csv
    val_datasets = DriverDataset(data_root, "Test_data_list.csv", transform=data_transforms['train'])
    print("train: {}, test: {}".format(len(train_datasets), len(val_datasets)))

    train_dataloaders = data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataloaders = data.DataLoader(val_datasets, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print("train dataloader: {}, val dataloader: {}".format(len(train_dataloaders), len(val_dataloaders)))

    if args.debug:
        x, y = next(iter(train_dataloaders))
        logger.append([x, y])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True

    num_classes = 5

    if  'resnet50' == args.model:
        my_model = models.resnet50(pretrained=True, num_classes=num_classes)
    elif 'resnet18_fdan' == args.model:
        my_model = resnet_fdan.resnet18_fdan(pretrained=True, num_classes=num_classes)
    elif 'resnet34_fdan' == args.model:
        my_model = resnet_fdan.resnet34_fdan(pretrained=True, num_classes=num_classes)
    elif 'resnet50_fdan' == args.model:
        my_model = resnet_fdan.resnet50_fdan(pretrained=True, num_classes=num_classes)
    elif 'resnet101_fdan' == args.model:
        my_model = resnet_fdan.resnet101_fdan(pretrained=True, num_classes=num_classes)
    elif 'resnet152_fdan' == args.model:
        my_model = resnet_fdan.resnet152_fdan(pretrained=True, num_classes=num_classes)
    elif 'resnet18' == args.model:
        my_model = resnet.resnet18(pretrained=True, num_classes=num_classes)
    elif 'resnet34' == args.model:
        my_model = resnet.resnet34(pretrained=True, num_classes=num_classes)
    elif 'alexnet' == args.model:
        my_model = models.alexnet(pretrained=False, num_classes=num_classes)
    elif 'resnet18_adnet' == args.model:
        my_model = resnet_adnet.resnet18_adnet(pretrained=True, num_classes=num_classes)
    elif 'resnet34_adnet' == args.model:
        my_model = resnet_adnet.resnet34_adnet(pretrained=True, num_classes=num_classes)
    elif 'resnet18_se' == args.model:
        my_model = resnet_se.resnet18_se(pretrained=True, num_classes=num_classes)
    elif 'fcnn18' == args.model:
        my_model = fcnn.fcnn18(pretrained=True, num_classes=num_classes)
    elif 'tscnn' == args.model:
        my_model = tscnn.resnet18(pretrained=True, num_classes=num_classes)
    elif 'resnet18_GMM' == args.model:
        my_model = resnet_GMM.resnet18(pretrained=True, num_classes=num_classes)
    else:
        raise ModuleNotFoundError

    #print(my_model)
    print(sum([param.nelement() for param in my_model.parameters()]))
    #my_model.apply(fc_init)
    if is_use_cuda and 1 == len(gpus):
        my_model = my_model.cuda()
    elif is_use_cuda and 1 < len(gpus):
        my_model = nn.DataParallel(my_model.cuda())

    loss_fn = [nn.CrossEntropyLoss()]
    # Adjust the parameters of lr and momentum to train the models
    optimizer = optim.SGD(my_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)           #

    metric = [ClassErrorMeter([1,3], True)]
    start_epoch = 0
    num_epochs = args.num_epochs

    my_trainer = Trainer(my_model, args.model, loss_fn, optimizer, lr_schedule, 500, is_use_cuda, train_dataloaders, \
                        val_dataloaders, metric, start_epoch, num_epochs, args.debug, logger, writer)
    my_trainer.fit()
    logger.append('Optimize Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help='trainer debug flag')
    parser.add_argument('-g', '--gpu', default='0', type=str,
                        help='GPU ID Select')                    
    parser.add_argument('-d', '--data_root', default='./datasets',
                         type=str, help='data root')
    parser.add_argument('-t', '--train_file', default='./datasets/train.txt',
                         type=str, help='train file')
    parser.add_argument('-v', '--val_file', default='./datasets/val.txt',
                         type=str, help='validation file')
    parser.add_argument('-m', '--model', default='resnet18_fdan',
                         type=str, help='model type')
    parser.add_argument('--batch_size', default=64,
                         type=int, help='model train batch size')
    parser.add_argument('--num_epochs', default=60,
                         type=int, help='number of epochs')
    parser.add_argument('--display', action='store_true', dest='display',
                        help='Use TensorboardX to Display')
    args = parser.parse_args()

    main(args)
