# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2020/3/2
# @institute: UMSI
# @version: 0.1 alpha

# from unet_model import UNet
from two_layer_unet_model import UNet as UNet2
from unet_model4 import UNet as UNet4
from unet_model3 import UNet as UNet3
import logging
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import time
from data_ldr import *
from utils_predict import *
# Argument
# num_epochs = 20
# learning_rate = 0.
PENALTY = 1.01
REWARD = 0.99


Weight = np.full((3,3),1/3)  # Weight[0,1,2] for small mid large; Weight[][0, 1, 2] for 2, 3, 4 layer.

# Data Loader
dataset_train_large = img_seg_ldr(data_dir = "./large")
train_loader_large = DataLoader(dataset_train_large, batch_size=1, shuffle=True)

dataset_train_mid = img_seg_ldr(data_dir = "./mid")
train_loader_mid = DataLoader(dataset_train_mid, batch_size=1, shuffle=True)

dataset_train_small = img_seg_ldr(data_dir = "./small")
train_loader_small = DataLoader(dataset_train_small, batch_size=1, shuffle=True)


# Device identification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Try to find out if the computer have a CUDA with Nivida GPU, else we will use CPU to work

# Model
net_2layer = UNet2(n_channels=3, n_classes=4).to(device)
net_2layer.load_state_dict(torch.load('./model/two_layer_CP_epoch15.pth', map_location=device))

net_3layer = UNet3(n_channels=3, n_classes=4).to(device)
net_3layer.load_state_dict(torch.load('./model/model14.pth', map_location=device))

net_4layer = UNet4(n_channels=3, n_classes=4).to(device)
net_4layer.load_state_dict(torch.load('./model/CP_epoch30.pth', map_location=device))
# net_4layer.load_state_dict(torch.load('./model/CP_epoch30.pth'))


# Loss Function 
criterion = nn.CrossEntropyLoss(weight =torch.from_numpy(np.array([1,2,2,1])).type(torch.FloatTensor).to(device), reduction='sum')


net_2layer.eval()
net_3layer.eval()
net_4layer.eval()
#For small images
print("=================== Start on small images =========================")
for i, (images, labels) in enumerate(train_loader_small):
    print("image{}:".format(i))

    images = images.to(device)
    labels = labels.to(device)
    loss = []

    outputs_2 = net_2layer(images)
    loss_2 = criterion(outputs_2, labels)
    loss.append(loss_2)
    print("2 layer loss: {}".format(loss_2.item()))


    outputs_3 = net_3layer(images)
    loss_3 = criterion(outputs_3, labels)
    loss.append(loss_3)
    print("3 layer loss: {}".format(loss_3.item()))

    outputs_4 = net_4layer(images)
    loss_4 = criterion(outputs_4, labels)
    loss.append(loss_4)
    print("4 layer loss: {}".format(loss_4.item()))

    index_max = loss.index(max(loss))
    index_min = loss.index(min(loss))

    Weight[0][index_max] *= REWARD
    Weight[0][index_min] *= PENALTY

    print("Weight:")
    print(Weight)


np.save('weightsmall.npy', Weight)
#For mid images
print("=================== Start on mid images =========================")
for i, (images, labels) in enumerate(train_loader_mid):
    print("image{}:".format(i))
    images = images.to(device)
    labels = labels.to(device)
    loss = []

    outputs_2 = net_2layer(images)
    loss_2 = criterion(outputs_2, labels)
    loss.append(loss_2)
    print("2 layer loss: {}".format(loss_2.item()))


    outputs_3 = net_3layer(images)
    loss_3 = criterion(outputs_3, labels)
    loss.append(loss_3)
    print("3 layer loss: {}".format(loss_3.item()))

    outputs_4 = net_4layer(images)
    loss_4 = criterion(outputs_4, labels)
    loss.append(loss_4)
    print("4 layer loss: {}".format(loss_4.item()))

    index_max = loss.index(max(loss))
    index_min = loss.index(min(loss))

    Weight[1][index_max] *= REWARD
    Weight[1][index_min] *= PENALTY

    print("Weight:")
    print(Weight)    

np.save('weightmid.npy', Weight)
#For large images
print("=================== Start on large images =========================")
for i, (images, labels) in enumerate(train_loader_large):
    print("image{}:".format(i))
    images = images.to(device)
    labels = labels.to(device)
    loss = []

    outputs_2 = net_2layer(images)
    loss_2 = criterion(outputs_2, labels)
    loss.append(loss_2)
    print("2 layer loss: {}".format(loss_2.item()))


    outputs_3 = net_3layer(images)
    loss_3 = criterion(outputs_3, labels)
    loss.append(loss_3)
    print("3 layer loss: {}".format(loss_3.item()))

    outputs_4 = net_4layer(images)
    loss_4 = criterion(outputs_4, labels)
    loss.append(loss_4)
    print("4 layer loss: {}".format(loss_4.item()))

    index_max = loss.index(max(loss))
    index_min = loss.index(min(loss))

    Weight[2][index_max] *= REWARD
    Weight[2][index_min] *= PENALTY

    print("Weight:")
    print(Weight)


np.save('weight.npy', Weight)
