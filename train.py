# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2020/3/2
# @institute: UMSI
# @version: 0.1 alpha


import sys
sys.path.append('./model')
sys.path.append('./data')
from model.unet_model import UNet
from model.two_layer_unet_model import UNet2
from model.three_layer_unet_model import UNet3
from model.unetpp import UNetpp
import logging
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import time
from data.data_ldr import *
from evaluation.utils_predict import *
import argparse

parser = argparse.ArgumentParser(description='Status')
parser.add_argument('--epoch', type=int,help='the number of epoch you want to train',default=0)
parser.add_argument('--lr', type=float,help='learning rate', default=0.01)
parser.add_argument('--batchsize', type=int,help='batch size', default=1)
parser.add_argument('--model', type=str,help='the model you want to use, now we support unet, unetpp, unet2, unet3, resunet', default="unet")
parser.add_argument('--load', type=str,help='the pretrained model path you want to use', default=None)
parser.add_argument('--dir_train', type=str, help='training set directory', default="./image-example")
parser.add_argument('--dir_test', type=str, help='test set directory', default="./image-example")
args = parser.parse_args()

# Argument
num_epochs = args.epoch
learning_rate = args.lr
model = args.model
load_model = args.load
dir_train = args.dir_train
dir_test = args.dir_test
bs = args.batchsize
# Data Loader
dataset_train = img_seg_ldr(data_dir=dir_train)
train_loader = DataLoader(dataset_train, batch_size=bs, shuffle=True)
dataset_test = img_seg_ldr(data_dir=dir_test)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True)
# Device identification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Try to find out if the computer have a CUDA with Nivida GPU, else we will use CPU to work

# Model
if model == "unet":
    net = UNet(n_channels=3, n_classes=4).to(device)
if model == "unet3":
    net = UNet3(n_channels=3, n_classes=4).to(device)
if model == "unet2":
    net = UNet2(n_channels=3, n_classes=4).to(device)
if model == "resunet":
    net = Unet_Resnet(in_channels=3).to(device)
if model == "unetpp":
    net = UNetpp(in_ch=3, out_ch=4).to(device)

# Loss Function 
criterion = nn.CrossEntropyLoss(weight =torch.from_numpy(np.array([1,2,2,1])).type(torch.FloatTensor).to(device), reduction='sum')

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Save model:
dir_checkpoint = 'checkpoints/'

# Read from a current trained model:
if not load_model is None:
    try:
        model_path = load_model
        net.load_state_dict(torch.load(model_path))
    except:
        print("We don't have the current trained model for {}".format(model_path))

# Training
net.train()
total_step = len(train_loader)
for epoch in range(0, num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 1 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                .format(epoch, num_epochs, i+1, total_step, loss.item()))
    try:
        os.mkdir(dir_checkpoint)
        logging.info('Created checkpoint directory')
    except OSError:
        pass
    torch.save(net.state_dict(),dir_checkpoint + f'three_layer_CP_epoch{epoch}.pth')
    logging.info(f'Checkpoint {epoch} saved !')

net.eval()
for i, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    loss = criterion(outputs, labels)
    print(loss.item())
    savepic(outputs[0,:], "{}_predict_three_layer.png".format(i))