## Tips
- Any problem, free to contact the authors via emails: whchen@hnu.edu.cn or xgqman@hnu.edu.cn.   
- Do not post issues with GitHub as much as possible, just in case that I could not receive the emails from github thus ignore the posted issues.  
- The paper of FDAN: Fuzzy Deep Attention Networks for Driver Behavior Recognition is being reviewed in Journal of Systems Architecture.

## Requirements
- python >=3.6
- torch >=1.9.0
- tensorboardX >=2.10.1
- numpy >=1.19.5
- matplotlib >=3.3.4
- opencv >=4.6.0
- torchvision >=0.10.0
- cuda >=11.5

## Pre-train Models
|[ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) | [ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) |[ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | [ResNet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth)|| [Alexnet](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth)|

## Datasets
- data_AUC, from the American University in Cairo
- data_HNU, from Hunan University

## Training
- We train models in the paper on the HNU and AUC datasets, the trained models are stored in .ckpt format.  
- The dataset path: in the train.py file, set data_ Root='/ Data_ AUC/' if the experiment uses the AUC dataset; set data_ Root='/ Data_ HNU/' if the experiment uses the HNU dataset.
- The models: resnet18, alexnet, resnet18_se, resnet18_fdan, resnet18_adnet, fcnn18, resnet18_GMM, tscnn
- We can use the following commands to train a model.  
 i.e., python train.py --model resnet18_fdan --batch_size 64 --num_epochs 60 --display  
The model name, batch size and the number of epochs can be set according to the training requirements.
- If the models are trained using leaving-N-driver-out cross validation, the files Train_data_list_N.csv and Test_data_list_N.csv are selected as the training and validation sets, respectively.

## Test



