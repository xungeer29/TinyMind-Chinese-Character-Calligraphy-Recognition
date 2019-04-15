# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

"""
qi=1-smoothing(if i=y)
qi=smoothing / (self.size - 1) (otherwise)#所以默认可以fill这个数，只在i=y的地方执行1-smoothing
另外KLDivLoss和crossentroy的不同是前者有一个常数
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.9, 0.2, 0.1, 0], 
                             [1, 0.2, 0.7, 0.1, 0]])
对应的label为
tensor([[ 0.0250,  0.0250,  0.9000,  0.0250,  0.0250],
        [ 0.9000,  0.0250,  0.0250,  0.0250,  0.0250],
        [ 0.0250,  0.0250,  0.0250,  0.9000,  0.0250]])
区别于one-hot的
tensor([[ 0.,  0.,  1.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.]])
"""
class LabelSmoothing(nn.Module):
    "Implement label smoothing.  size表示类别总数"
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        #self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing # if i=y 的公式
        self.smoothing = smoothing
        self.smoothed_label = None
        
    def forward(self, x, target):
        """
        x: 网络输出
        target表示label（M，）
        """
        smoothed_label = x.data.clone()
        num_classes = x.size(1)
        smoothed_label.fill_(self.smoothing / (num_classes - 1)) # otherwise的公式
        #target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        smoothed_label.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return smoothed_label

if __name__=="__main__":
    # Example of label smoothing. 
    import sys
    sys.path.append('/home/gfx/Projects/Tinymind')
    import torchvision.models as models
    from networks.network import *
    from config import config

    backbone = models.resnet18(pretrained=True)
    models = ResNet18(backbone, 100)
    # print models
    data = torch.randn(8, 1, 128, 128)
    x = models(data)
    print(x)

    labelsmooth = LabelSmoothing(smoothing= 0.1)
    smoothed_label = labelsmooth(x, Variable(torch.LongTensor([2, 0, 10, 90, 1, 2, 4, 6])))
    print smoothed_label

