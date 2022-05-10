# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:53:53 2018

@author: wuyz
"""
import os
import cv2
import numpy as np
import core_lzj


class ImageDeal:
    def __init__(self, height=128, weight=128, bk_cut=0.0, mp=1.0):
        self.height = height
        self.weight = weight
        self.h_num = 55
        self.w_num = 55
        self.bk_cut = bk_cut
        self.mp = mp
        self.img_data = 0

    def get_h_w_num(self, img_data):
        self.img_data = img_data
        self.h_num = int(img_data.shape[0] / self.height)
        self.w_num = int(img_data.shape[1] / self.weight)

    def num_white_points(self, img_data):
        num_temp = 0
        for i in range(self.weight):
            for j in range(self.height):
                if img_data[i, j, 0] == 255 and img_data[i, j, 0] == 255 and img_data[i, j, 0] == 255:
                    num_temp += 1
        return num_temp

    def image_cut(self, path):

        n = 0
        # im_max=np.max(img)
        im_mean = np.mean(self.img_data)
        im_points = self.weight * self.height
        print('im_mean=', im_mean)
        for i in range(self.h_num):
            for j in range(self.w_num):
                ix1 = self.height * i
                ix2 = ix1 + self.height
                jy1 = self.weight * j
                jy2 = jy1 + self.weight
                im_temp = self.img_data[ix1:ix2, jy1:jy2, :]
                name = path + '/' + path.split('/')[-1] + '_' + str(i + 1).zfill(2) + '_' + str(j + 1).zfill(2) + '.png'

                nwp = self.num_white_points(im_temp)
                if np.mean(im_temp) >= self.bk_cut * im_mean and nwp <= self.mp * im_points:
                    cv2.imwrite(name, im_temp)
                    n += 1
        print('num_pictures=', n)


# print(im.shape)
# print(im[0,39,0])
if __name__ == '__main__':
    img_h = 300
    img_w = 300
    threshold = 0.0
    # num = 0
    im = ImageDeal(height=img_h, weight=img_w, bk_cut=threshold)
    img_path = 'probability/186z.tif'
    img_temp = cv2.imread(img_path)
    im.get_h_w_num(img_data=img_temp)
    # dest_path = './check/145x_' + im.h_num.__str__() + '_' + im.w_num.__str__()
    dest_path = 'probability/186z300_' + im.h_num.__str__() + '_' + im.w_num.__str__()
    core_lzj.check_folder_existence(dest_path)
    im.image_cut(path=dest_path)
