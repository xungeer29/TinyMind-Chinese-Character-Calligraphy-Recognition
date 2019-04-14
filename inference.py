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
    # label2name
    label2name = read_label('./data/label_list.txt')
    
    # TTA raw invert CenterCrop
    augments = [0, 1, 2]
    tta0, tta1, tta2 = {}, {}, {}
    for idx in range(len(augments)):
        print('TTA {}'.format(idx))
        dst_test = TMTestDataset('./data/test2.txt', width=config.width, height=config.height, 
                                 transform=transform, augment=augments[idx])
        dataloader_valid = DataLoader(dst_test, shuffle=False, batch_size=config.batch_size/2, 
                                      num_workers=config.num_workers)

        sum = 0
        model.eval()
        results = []
        probs_all = []
        for ims, im_names in tqdm(dataloader_valid):
            input = Variable(ims).cuda()
            output = model(input)

            _, preds = output.topk(5, 1, True, True)
            preds = preds.cpu().detach().numpy()
            for pred, im_name in zip(preds, im_names):
                top5_name = [label2name[p] for p in pred]
                results.append({'filename':im_name, 'label':''.join(top5_name)})

            # TTA
            probs = F.softmax(output)
            probs = probs.cpu().detach().numpy()
            for prob, im_name in zip(probs, im_names):
                if idx == 0:
                    tta0[im_name] = prob
                elif idx == 1:
                    tta1[im_name] = prob
                elif idx == 2:
                    tta2[im_name] = prob
                else:
                    print('Error: No  other TTA method!!!')
                    break
        # save no TTA
        df = pd.DataFrame(results, columns=['filename', 'label'])
        df.to_csv('./data/result_no_TTA.csv', index=False)
    # endemble TTA
    #print tta0
    tta_results = []
    for key in tta0.keys():
        prob = (tta0[key] + tta1[key] + tta2[key]) / 3.
        top5_idx = prob.argsort()[-5:][::-1]
        top5_name = [label2name[p] for p in top5_idx]
        tta_results.append({'filename':key, 'label':''.join(top5_name)})
    # save TTA result
    tta_df = pd.DataFrame(tta_results, columns=['filename', 'label'])
    tta_df.to_csv('./data/result_3TTA.csv', index=False)

if __name__ == '__main__':
    inference()
