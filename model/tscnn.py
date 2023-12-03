import torch
import torch.nn as nn
from torchvision import models

# 定义VGG16模型
def create_vgg16():
    vgg16_model = models.vgg16(pretrained=True)
    # 为了防止权重被更新，将VGG16模型的所有层设为不可训练
    for param in vgg16_model.parameters():
        param.requires_grad = False
    return vgg16_model

# 定义两个VGG16模型
vgg16_model_1 = create_vgg16()
vgg16_model_2 = create_vgg16()

# 定义自定义的分类网络
class CustomClassifier(nn.Module):
    def __init__(self):
        super(CustomClassifier, self).__init__()
        # 添加卷积层
        self.conv1 = nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # 添加池化层
        self.pooling = nn.MaxPool2d(kernel_size=7, stride=7)
        # 添加全连接层
        self.fc = nn.Linear(512, 5)  # 输入大小根据级联后的特征图大小调整

    def forward(self, x):
        # 卷积操作
        x = self.conv1(x)
        x = self.conv2(x)
        # 池化操作
        x = self.pooling(x)
        # 展平操作
        x = x.view(x.size(0), -1)
        #print(x.shape)
        # 全连接层
        x = self.fc(x)
        return x

# 实例化自定义分类网络
#custom_classifier = CustomClassifier(num_classes)

# 将两个VGG16模型输出的特征图级联
class TwoStreamVGG16(nn.Module):
    def __init__(self, vgg16_model_1, vgg16_model_2, custom_classifier, num_classes):
        super(TwoStreamVGG16, self).__init__()
        self.features_1 = vgg16_model_1.features
        self.features_2 = vgg16_model_2.features
        self.classifier = custom_classifier
        self.conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)

    def forward(self, x1):    #, x2):
        # 分别通过两个VGG16模型
        x11 = self.features_1(x1)
        x11 = self.conv(x11)
        #print(x11.shape)
        #x2 = self.features_2(x2)
        x22 = self.features_2(x1)
        x22 = self.conv(x22)
        #print(x22.shape)
        # 将特征图级联
        x = torch.cat((x11, x22), dim=1)
        # 通过自定义分类器
        x = self.classifier(x)
        #print(x.shape)
        return x

# 实例化两路VGG16网络
#two_stream_vgg16 = TwoStreamVGG16(vgg16_model_1, vgg16_model_2, CustomClassifier(num_classes))

def tscnn16(pretrained=False, **kwargs):
    """Constructs a tscnn model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TwoStreamVGG16(create_vgg16(), create_vgg16(), CustomClassifier(), **kwargs)
    return model

# 打印模型结构
#print(two_stream_vgg16)
