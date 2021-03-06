# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/gfx/Projects/Tinymind')
import os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

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
from utils.label_smooth import *


def train():
    # model
    if config.model == 'ResNet18':
        backbone = models.resnet18(pretrained=True)
        model = ResNet18(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet34':
        backbone = models.resnet34(pretrained=True)
        model = ResNet34(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet50':
        backbone = models.resnet50(pretrained=True)
        model = ResNet50(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet101':
        backbone = models.resnet101(pretrained=True)
        model = ResNet101(backbone, num_classes=config.num_classes)
    elif config.model == 'ResNet152':
        backbone = models.resnet152(pretrained=True)
        model = ResNet152(backbone, num_classes=config.num_classes)
    else:
        print('ERROR: No model {}!!!'.format(config.model))
    print model
    # model = torch.nn.DataParallel(model)
    model.cuda()
    
    # freeze layers
    if config.freeze:
        for p in model.backbone.layer1.parameters(): p.requires_grad = False
        for p in model.backbone.layer2.parameters(): p.requires_grad = False
        for p in model.backbone.layer3.parameters(): p.requires_grad = False
        #for p in model.backbone.layer4.parameters(): p.requires_grad = False


    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    # label smoothing
    label_smoothing = LabelSmoothing(smoothing=0.05)

    # train data
    transform = transforms.Compose([transforms.Scale(200),
                                    transforms.RandomSizedCrop(180),
                                    transforms.RandomRotation(10),
                                    transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_train = TMDataset('./data/train.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=config.batch_size, 
                                  num_workers=config.num_workers)

    # validation data
    transform = transforms.Compose([transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_valid = TMDataset('./data/valid.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=config.batch_size/2, num_workers=config.num_workers)

    # log
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log = open('./log/log.txt', 'a')

    log.write('-'*30+'\n')
    log.write('model:{}\nnum_classes:{}\nnum_epoch:{}\nlearning_rate:{}\nim_width:{}\nim_height:{}\niter_smooth:{}\nOHEM:{}\nOHEM_ratio:{}\n'.format(
               config.model, config.num_classes, config.num_epochs, config.lr, 
               config.width, config.height, config.iter_smooth, config.OHEM, config.OHEM_ratio))

    # load checkpoint
    if config.resume:
        print 'resume checkpoint...'
        model = torch.load(os.path.join('./checkpoints', config.checkpoint))

    if not os.path.exists('./figs'):
        os.makedirs('./figs')

    # train
    sum = 0
    train_loss_sum = 0
    train_top1_sum = 0
    train_top5_sum = 0
    max_val_top1_acc = 0
    max_val_top5_acc = 0
    train_draw_acc = []
    val_draw_acc = []
    for epoch in range(config.num_epochs):
        ep_start = time.time()

        # adjust lr
        # lr = half_lr(config.lr, epoch)
        lr = step_lr(epoch)

        # optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                     lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)

        model.train()
        top1_sum = 0
        for i, (ims, label) in enumerate(dataloader_train):
            input = Variable(ims).cuda()
            target = Variable(label).cuda().long()

            output = model(input)

            if config.smooth_label:
                target = label_smoothing(output, target)
            
            # OHEM: online hard example mining
            if not config.OHEM:
                loss = criterion(output, target)
            elif config.OHEM:
                if epoch < 50:
                    loss = criterion(output, target)
                else:
                    loss = F.cross_entropy(output, target, reduce=False)
                    OHEM, _ = loss.topk(int(config.num_classes*config.OHEM_ratio), dim=0, 
                                        largest=True, sorted=True)
                    loss = OHEM.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(output.data, target.data, topk=(1, 5))
            train_loss_sum += loss.data.cpu().numpy()
            train_top1_sum += acc[0]
            train_top5_sum += acc[1]
            sum += 1
            top1_sum += acc[0]

            if (i+1) % config.iter_smooth == 0:
                print('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f, top5: %.4f'
                       %(epoch+1, config.num_epochs, i+1, len(dst_train)//config.batch_size, 
                       lr, train_loss_sum/sum, train_top1_sum/sum, train_top5_sum/sum))
                log.write('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f, top1: %.4f, top5: %.4f\n'
                           %(epoch+1, config.num_epochs, i+1, len(dst_train)//config.batch_size, 
                           lr, train_loss_sum/sum, train_top1_sum/sum, train_top5_sum/sum))
                sum = 0
                train_loss_sum = 0
                train_top1_sum = 0
                train_top5_sum = 0

        train_draw_acc.append(top1_sum/len(dataloader_train))
        
        epoch_time = (time.time() - ep_start) / 60.
        if epoch % 1 == 0 and epoch < config.num_epochs:
            # eval
            val_time_start = time.time()
            val_loss, val_top1, val_top5 = eval(model, dataloader_valid, criterion)
            val_draw_acc.append(val_top1)
            val_time = (time.time() - val_time_start) / 60.

            print('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, Val_top5: %.4f'
                   %(epoch+1, config.num_epochs, val_loss, val_top1, val_top5))
            print('epoch time: {}s'.format(epoch_time*60))
            if val_top1[0].data > max_val_top1_acc:
                max_val_top1_acc = val_top1[0].data
                print('Taking top1 snapshot...')
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save(model, '{}/{}_top1.pth'.format('checkpoints', config.model))

            if val_top5[0].data > max_val_top5_acc:
                max_val_top5_acc = val_top5[0].data
                print('Taking top5 snapshot...')
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save(model, '{}/{}_top5.pth'.format('checkpoints', config.model))
            log.write('Epoch [%d/%d], Val_Loss: %.4f, Val_top1: %.4f, Val_top5: %.4f\n'
                       %(epoch+1, config.num_epochs, val_loss, val_top1, val_top5))

        draw_curve(train_draw_acc, val_draw_acc)
    log.write('-'*30+'\n')
    log.close()

# validation
def eval(model, dataloader_valid, criterion):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    val_top5_sum = 0
    model.eval()
    for ims, label in dataloader_valid:
        input_val = Variable(ims).cuda()
        target_val = Variable(label).cuda()
        output_val = model(input_val)
        loss = criterion(output_val, target_val)
        acc_val = accuracy(output_val.data, target_val.data, topk=(1, 5))
        
        sum += 1
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += acc_val[0]
        val_top5_sum += acc_val[1]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum
    avg_top5 = val_top5_sum / sum
    return avg_loss, avg_top1, avg_top5

if __name__ == '__main__':
    train()
