# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2019/3/15
# @institute:JI@SJTU,UMSI
# @version: 0.1 alpha

import numpy as np
import os
import csv
from config import *

def content_generator(data_dir = ".", save_dir = ".", new = True):

    print("Generating new content...")

    image_dir = ""
    mask_dir = ""
    image_list = []
    mask_list = []

    path = os.listdir(data_dir)
    for p in path:
        p_comb = os.path.join(data_dir, p)
        if os.path.isdir(p_comb):
            if p == "image":
                image_dir = os.path.join(data_dir, p)
            if p == "mask":
                mask_dir = os.path.join(data_dir, p)
    
    assert image_dir != "" and mask_dir != "", "you should pass a data_dir which has the \"image\" and \"mask\" folder."
    image_fileset = set(os.listdir(os.path.join(data_dir, "image")))
    mask_fileset = set(os.listdir(os.path.join(data_dir, "mask")))

    for filename in image_fileset:
        file_wo_ext = os.path.splitext(filename)[0]
        mask_filename = file_wo_ext + "-mask" + os.path.splitext(filename)[-1]
        if not os.path.isfile(os.path.join(os.path.join(data_dir, "mask"),mask_filename)):
            continue
        image_list.append(os.path.join(os.path.join(data_dir, "image"),filename))
        mask_list.append(os.path.join(os.path.join(data_dir, "mask"),mask_filename))

    file_num = len(image_list)
    print("Training data image number: {}".format(file_num))
    print("Saving the content to: {}".format(os.path.join(save_dir,'content.csv')))

    with open(os.path.join(save_dir,'content.csv'),'w',newline = "") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["image", "mask"])
        for i in range(file_num):
            f_csv.writerow([image_list[i], mask_list[i]])
    
    return image_list, mask_list

def pixelClassify(pixel):
    '''This function is used to match the RGB color of a pixel to a class'''
    # 0 - background
    # 1 - inflammatory cells
    # 2 - nuclei
    # 3 - cytoplasm
    red = pixel[0]
    for i in range(len(COLOR_DICT)):
        if red == COLOR_DICT[i]:
            return i
    return 0


def generateLabel(maskImage):
    '''This function is used to generate a one-hot encoded image for mask image
       UPDATE: return mask is a labeled mask rather than a one-hot encoded image'''
    maskImage = np.asarray(maskImage)
    SIZE = maskImage.shape[0]
    # one_hot_mask = np.zeros((4, SIZE,SIZE))
    one_hot_mask = np.zeros((SIZE,SIZE))
    for i in range(SIZE):
        for j in range(SIZE):
            # one_hot_mask[pixelClassify(maskImage[i,j,:]), i, j] = 1 # for one-hot open when using MSE/SMOOTHL1
            one_hot_mask[i, j] = pixelClassify(maskImage[i,j,:]) # for 0 <= one_hot_mask[i, j] <= (C-1) open when using NLL/CROSSENTROPY
    return one_hot_mask