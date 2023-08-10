
## Introduction
The codes are [PyTorch](https://pytorch.org/) re-implement version for paper: FDAN, but the paper is under review in Journal of Systems Architecture.



## Requirements
- Python3
- PyTorch 0.4.1
- tensorboardX (optional)
- torchnet
- pretrainedmodels (optional)

## Results
We train models in the paper on the HNU and AUC dataset.

## Train command
python train.py --model <model name> --batch_size <the batch size> --num_epochs <the number of epochs> --display

##models
resnet18_fdan
resnet18
resnet18_adnet
alexnet

## todo
