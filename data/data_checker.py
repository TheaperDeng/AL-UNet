#!usr/bin/python3 
#@author:Deng Junwei
#coding:utf-8
#@insitute:JI@SJTU,UMSI

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_ldr import *

dataset_train = img_seg_ldr(data_dir="../image-example/")
print('dataset_train is sucessfully loaded')
train_loader = DataLoader(dataset_train, batch_size=5, shuffle=True)
for i, (images, labels) in enumerate(train_loader):
    print('The ', i, 'batch:')
    print('image size: ', str(images.shape))
    print('label size: ', str(labels.shape))
    if i >= 4:
        break

