# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2019/3/15
# @institute:JI@SJTU,UMSI
# @version: 0.1 alpha

import numpy as np
import torchvision.transforms as transforms

# COLOR_DICT, you may change this if you need more or less color
Unlabelled  =   [0,     0,      0   ] #black, background
A           =   [255,   0,      80  ] #red, inflammatory cells 
B           =   [128,   255,    192 ] #Light blue, nuclei
C           =   [64,    255,    64  ] #green, cytoplasm 
COLOR_DICT  =   [Unlabelled[0],A[0],B[0],C[0]] #only R layer is used

data_transform = transforms.Compose([
        transforms.CenterCrop(500) ,
        transforms.ToTensor()
    ])