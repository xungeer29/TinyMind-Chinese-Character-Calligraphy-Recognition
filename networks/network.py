# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def whitening(im):
    batch_size, channel, h, w = im.shape
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im = torch.cat([(im[:,[0]]-mean[0])/std[0],
                    (im[:,[1]]-mean[1])/std[1],
                    (im[:,[2]]-mean[2])/std[2]], 1)
    return im

def l2_norm(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x

class ResNet18(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet18, self).__init__()
        self.backbone = model

        self.conv1 = nn.Conv2d(1, 16, 3) # 1 channel -> 3 channel
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(18432, 4608)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4608, num_classes)

    def forward(self, x):
        # x = whitening(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class ResNet34(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet34, self).__init__()
        self.backbone = model

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class ResNet50(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet50, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(8192, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)


    def forward(self, x):
        #x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResNet101(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet101, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(8192, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        #x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class ResNet152(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet152, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(8192, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)


    def forward(self, x):
        #x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == '__main__':
    backbone = models.resnet18(pretrained=True)
    models = ResNet18(backbone, 100)
    print models
    data = torch.randn(1, 1, 128, 128)
    x = models(data)
    #print(x)
    print(x.size())
