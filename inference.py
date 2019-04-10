# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/Tinymind')
import os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.nn.parallel.data_parallel import data_parallel
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset.dataset import *
from networks.network import *
from networks.lr_schedule import *
from metrics.metric import *
from utils.plot import *
from config import config

def read_label(path):
    data = open(path, 'r')
    label2name = {}
    for line in data.readlines():
        name, label = line.strip().split(' ')
        label2name[int(label)] = name
    return label2name

def inference():
    # model
    # load checkpoint
    model = torch.load(os.path.join('./checkpoints', config.checkpoint))
    # print model
    # model = torch.nn.DataParallel(model)
    model.cuda()
    
    # validation data
    transform = transforms.Compose([transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_test = TMTestDataset('./data/test2.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_valid = DataLoader(dst_test, shuffle=False, batch_size=config.batch_size/2, num_workers=config.num_workers)

    # label2name
    label2name = read_label('./data/label_list.txt')

    sum = 0
    model.eval()
    results = []
    for ims, im_names in tqdm(dataloader_valid):
        input = Variable(ims).cuda()
        output = model(input)

        _, preds = output.topk(5, 1, True, True)
        preds = preds.cpu().detach().numpy()
        for pred, im_name in zip(preds, im_names):
            top5_name = [label2name[p] for p in pred]
            results.append({'filename':im_name, 'label':''.join(top5_name)})
    df = pd.DataFrame(results, columns=['filename', 'label'])
    df.to_csv('./data/result.csv', index=False)

if __name__ == '__main__':
    inference()
