import torch
import torch.nn as nn
from torchvision import models

# ����VGG16ģ��
def create_vgg16():
    vgg16_model = models.vgg16(pretrained=True)
    # Ϊ�˷�ֹȨ�ر����£���VGG16ģ�͵����в���Ϊ����ѵ��
    for param in vgg16_model.parameters():
        param.requires_grad = False
    return vgg16_model

# ��������VGG16ģ��
vgg16_model_1 = create_vgg16()
vgg16_model_2 = create_vgg16()

# �����Զ���ķ�������
class CustomClassifier(nn.Module):
    def __init__(self):
        super(CustomClassifier, self).__init__()
        # ��Ӿ����
        self.conv1 = nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # ��ӳػ���
        self.pooling = nn.MaxPool2d(kernel_size=7, stride=7)
        # ���ȫ���Ӳ�
        self.fc = nn.Linear(512, 5)  # �����С���ݼ����������ͼ��С����

    def forward(self, x):
        # �������
        x = self.conv1(x)
        x = self.conv2(x)
        # �ػ�����
        x = self.pooling(x)
        # չƽ����
        x = x.view(x.size(0), -1)
        #print(x.shape)
        # ȫ���Ӳ�
        x = self.fc(x)
        return x

# ʵ�����Զ����������
#custom_classifier = CustomClassifier(num_classes)

# ������VGG16ģ�����������ͼ����
class TwoStreamVGG16(nn.Module):
    def __init__(self, vgg16_model_1, vgg16_model_2, custom_classifier, num_classes):
        super(TwoStreamVGG16, self).__init__()
        self.features_1 = vgg16_model_1.features
        self.features_2 = vgg16_model_2.features
        self.classifier = custom_classifier
        self.conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)

    def forward(self, x1):    #, x2):
        # �ֱ�ͨ������VGG16ģ��
        x11 = self.features_1(x1)
        x11 = self.conv(x11)
        #print(x11.shape)
        #x2 = self.features_2(x2)
        x22 = self.features_2(x1)
        x22 = self.conv(x22)
        #print(x22.shape)
        # ������ͼ����
        x = torch.cat((x11, x22), dim=1)
        # ͨ���Զ��������
        x = self.classifier(x)
        #print(x.shape)
        return x

# ʵ������·VGG16����
#two_stream_vgg16 = TwoStreamVGG16(vgg16_model_1, vgg16_model_2, CustomClassifier(num_classes))

def tscnn16(pretrained=False, **kwargs):
    """Constructs a tscnn model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TwoStreamVGG16(create_vgg16(), create_vgg16(), CustomClassifier(), **kwargs)
    return model

# ��ӡģ�ͽṹ
#print(two_stream_vgg16)
