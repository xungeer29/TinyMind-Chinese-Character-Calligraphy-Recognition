# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/Tinymind')

import os
import random
from config import config
from tqdm import tqdm

random.seed(config.seed)

if not os.path.exists('./data'):
    os.makedirs('./data')

train_txt = open('./data/train.txt', 'w')
val_txt = open('./data/valid.txt', 'w')
label_txt = open('./data/label_list.txt', 'w')

label_list = []

for dir in tqdm(os.listdir(os.path.join(config.data_root, 'train'))):
    if dir not in label_list:
        label_list.append(dir)
        label_txt.write('{} {}\n'.format(dir, str(len(label_list)-1)))
        data_path = os.path.join(config.data_root, 'train', dir)
        train_list = random.sample(os.listdir(data_path), 
                                   int(len(os.listdir(data_path))*0.975))
        for im in train_list:
            train_txt.write('{}/{}/{} {}\n'.format('train', dir, im, str(len(label_list)-1)))
        for im in os.listdir(data_path):
            if im in train_list:
                continue
            else:
                val_txt.write('{}/{}/{} {}\n'.format('train', dir, im, str(len(label_list)-1)))

# test1
test1_txt = open('./data/test1.txt', 'w')
for im in tqdm(os.listdir(os.path.join(config.data_root, 'test1'))):
    test1_txt.write('{}/{}\n'.format('test1', im))
# test2
test2_txt = open('./data/test2.txt', 'w')
for im in tqdm(os.listdir(os.path.join(config.data_root, 'test2'))):
    test2_txt.write('{}/{}\n'.format('test2', im))
