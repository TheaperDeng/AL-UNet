import os
from PIL import Image
import numpy as np
import scipy.misc
import cv2

ws = [6.73e-03, 5.23, 9.99e-01]
sumw = sum(ws)
for i in range(len(ws)):
    ws[i] = ws[i]/sumw
print(ws)
wm = [7.55e-02, 1.26e+01, 3.67e-02]
sumw = sum(wm)
for i in range(len(wm)):
    wm[i] = wm[i]/sumw
print(wm)
wl = [4.96e-04, 6.22, 1.09e+01]
sumw = sum(wl)
for i in range(len(wl)):
    wl[i] = wl[i]/sumw
print(wl)

def aggregator(image_list, weight, name):
    length = image_list[0].shape[0]
    width = image_list[0].shape[1]
    new_image = np.zeros([length,width],dtype=np.uint8)
    for image in image_list:
        for i in range(length):
            for j in range(width):
                # vote = 0
                for image in image_list:
                    if image[i][j][0] == 255 or image[i][j][0] == 128:
                        new_image[i][j] = 255
                        # vote += 1
                # if vote >= 2:
                #     new_image[i][j] = 255
    scipy.misc.toimage(new_image, cmin=0.0, cmax=255.0).save('outfile.jpg')
    _, labels = cv2.connectedComponents(new_image)
    count = 0
    for i in range(length):
        for j in range(width):
            if not labels[i][j] == 0:
                if labels[i][j] > count:
                    count = labels[i][j]
    # for i in range(length):
    #     for j in range(width):
    #         if not labels[i][j] == 0:
    #             new_image[i][j] = labels[i][j] * 255/count
    # scipy.misc.toimage(new_image, cmin=0.0, cmax=255.0).save('outfile2.jpg')
    res = []
    for k in range(1,count+1):
        count255 = 0
        count128 = 0
        for image_id in range(len(image_list)):
            for i in range(length):
                for j in range(width):
                    if labels[i][j] == k:
                        if image_list[image_id][i][j][0] == 255:
                            count255 += weight[image_id]
                        if image_list[image_id][i][j][0] == 128:
                            count128 += weight[image_id]
        if count255*0.7 > count128:
            res.append(255)
        else:
            res.append(128)
    new_image2 = np.zeros([length,width])
    for i in range(length):
        for j in range(width):
            if not labels[i][j] == 0:
                new_image2[i][j] = res[labels[i][j]-1]
    scipy.misc.toimage(new_image2, cmin=0.0, cmax=255.0).save('outfile3.jpg')
    new_image3 = np.zeros([length,width,3],dtype=np.uint8)
    for i in range(length):
        for j in range(width):
            for image in image_list:
                if image[i][j][0] == 64:
                    new_image3[i][j][0] = 64
                    new_image3[i][j][1] = 255
                    new_image3[i][j][2] = 64
    for i in range(length):
        for j in range(width):
            if not labels[i][j] == 0:
                new_image3[i][j][0] = res[labels[i][j]-1]
                if res[labels[i][j]-1] == 255:
                    new_image3[i][j][1] = 0
                    new_image3[i][j][2] = 80
                else:
                    new_image3[i][j][1] = 255
                    new_image3[i][j][2] = 192
    im = Image.fromarray(new_image3)
    im.save("{}.png".format(name))
    print(name)

path = "./unet-2l/small"
files = os.listdir(path)
s = []
for file in files:
    if not os.path.isdir(file):
        try:
            s.append(file.split("_")[0]+"_" +file.split("_")[1])
        except:
            pass
print(s)

for string in s:
    # I1 = Image.open(r'D:/Wechat_File/WeChat Files/wxid_s15s0hnhfk0q12/FileStorage/File/2020-04/L234/unet-2l/{}_predict.png'.format(string)) 
    # I1_array = np.array(I1)
    I2 = Image.open(r'D:/Wechat_File/WeChat Files/wxid_s15s0hnhfk0q12/FileStorage/File/2020-04/L234/unet-3l/{}_predict.png'.format(string)) 
    I2_array = np.array(I2)
    I3 = Image.open(r'D:/Wechat_File/WeChat Files/wxid_s15s0hnhfk0q12/FileStorage/File/2020-04/L234/unet-4l/{}_predict.png'.format(string)) 
    I3_array = np.array(I3)
    # image_list = [I1_array, I2_array, I3_array]
    image_list = [I2_array, I3_array]
    aggregator(image_list, ws[1:], string)

