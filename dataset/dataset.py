# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/Tinymind')
import os
import cv2
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageOps

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from config import config
from augmentation import MyGaussianBlur

def read_txt(path):
    ims, labels = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split(' ')
            ims.append(im)
            labels.append(int(label))
    return ims, labels

class TMDataset(Dataset):
    def __init__(self, txt_path, width=256, height=256, transform=None):
        self.ims, self.labels = read_txt(txt_path)
        self.width = width
        self.height = height
        self.transform = transform

    def __getitem__(self, index):
        im_path = self.ims[index]
        label = self.labels[index]
        im_path = os.path.join(config.data_root, im_path)
        im = Image.open(im_path).convert('L')
        #im = im.resize((self.width, self.height))
        if self.transform is not None:
            if random.random() < 0.5:
                im = ImageOps.invert(im)
            if random.random() < 0.5:
                im = im.filter(MyGaussianBlur(radius=5))
            im = self.transform(im)

        return im, label

    def __len__(self):
        return len(self.ims)

class TMTestDataset(Dataset):
    def __init__(self, txt_path, width=256, height=256, transform=None, augment=None):
        test_data = open(txt_path, 'r')
        self.ims = [line.strip() for line in test_data.readlines()]
        self.width = width
        self.height = height
        self.transform = transform
        self.augment = augment

    def __getitem__(self, index):
        im_path = self.ims[index]
        name = im_path.split('/')[1]
        im_path = os.path.join(config.data_root, im_path)
        im = Image.open(im_path).convert('L')
        #im = im.resize((self.width, self.height))
        if self.augment == 2:
            im = im
        # invert color
        if self.augment == 1:
            im = ImageOps.invert(im)
        # center crop
        if self.augment == 0:
            w, h = im.size
            im = im.crop((10, 10, w-10, h-10))

        if self.transform is not None:
            im = self.transform(im)

        return im, name

    def __len__(self):
        return len(self.ims)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dst_train = TMDataset('./data/train.txt', width=256, height=256, transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=1, num_workers=0)
    #for im, loc, cls in dataloader_train:
    for data in dataloader_train:
        print data
        #print loc, cls
    
