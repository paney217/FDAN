
## Introduction
The codes are [PyTorch](https://pytorch.org/) re-implement version for paper: FDAN: Fuzzy Deep Attention Networks for Driver Behavior Recognition.

## Requirements
- Python3
- PyTorch 0.4.1
- tensorboardX (optional)
- torchnet
- pretrainedmodels (optional)

## Results
We train models in the paper on the HNU and AUC dataset, the trained models are stored in .pkt format.

## Command
- python train.py --model <model name> --batch_size <the batch size> --num_epochs <the number of epochs> --display
- For example, python train.py --model resnet18_fdan --batch_size 64 --num_epochs 60 --display

## Models
- resnet18
- alexnet
- resnet18_se
- resnet18_fdan
- resnet18_adnet
- fcnn18
- resnet18_GMM
- tscnn

## Datasets
- data_AUC, from the American University in Cairo
- data_HNU, from Hunan University

- [1] Y. Abouelnaga, H. Eraqi, and M. Moustafa. "Real-time Distracted Driver Posture Classification". Neural Information Processing Systems (NIPS 2018), Workshop on Machine Learning for Intelligent Transportation Systems, Dec. 2018. https://arxiv.org/abs/1706.09498
- [2] H. Eraqi, Y. Abouelnaga, M. Saad, M. Moustafa, "Driver Distraction Identification with an Ensemble of Convolutional Neural Networks", Journal of Advanced Transportation, Machine Learning in Transportation (MLT) Issue, 2019. https://www.hindawi.com/journals/jat/2019/4125865/

## Notes
The paper of FDAN is being reviewed in Journal of Systems Architecture. 
- E-mail: whchen@hnu.edu.cn
