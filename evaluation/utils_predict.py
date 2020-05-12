# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2019/3/21
# @institute:SJTU

import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import math



def savepic(output, filename):
    output = output.cpu().detach().numpy()
    img = _pic_coloring(output)
    cv2.imwrite(filename, img) # pay attention to the /255
    print(img.shape)

def _pic_coloring(output):
    img = np.zeros((output.shape[1], output.shape[2], 3))
    Unlabelled  =   np.array([0,     0,      0   ]) #black, background
    A           =   np.array([80,    0,      255 ]) #red, inflammatory cells 
    B           =   np.array([192,   255,    128 ]) #Light blue, nuclei
    C           =   np.array([64,    255,    64  ]) #green, cytoplasm      
    D           =   np.array([255,   96,     192 ]) #useless?
    for i in range(output.shape[1]):
        for j in range(output.shape[2]):
            max = 0
            maxindex = 0
            for k in range(4):
                if output[k,i,j] > max:
                    max = output[k,i,j]
                    maxindex = k
            if maxindex == 0:
                img[i, j, :] = Unlabelled
                continue
            if maxindex == 1:
                img[i, j, :] = A
                continue
            if maxindex == 2:
                img[i, j, :] = B
                continue
            if maxindex == 3:
                img[i, j, :] = C
                continue
    return img