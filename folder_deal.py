# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:32:42 2018

@author: 1
"""
import os
import cv2
import shutil
from image_deal import ImageDeal
import core_lzj


def pick_img(file_path):
    """
    if folders are already exist, please rmdir (shutil.rmtree) them firstly
    """
    files = core_lzj.each_img(file_path)
    for file in files:
        if os.path.isdir(file):
            shutil.rmtree(file)
            # os.rmdir(file)
    """
    cv.imread(img) and cut it into correponding folder
    """
    # files = myfunctions.eachfile(filepath)
    for num, file in enumerate(files):
        img = cv2.imread(file)
        im.get_h_w_num(img_data=img)
        path = file_path + file.split('/')[-1].split('.')[0] + '_' + im.h_num.__str__() + '_' + im.w_num.__str__()
        print(path, file)
        core_lzj.check_folder_existence(path)
        im.image_cut(path)


if __name__ == "__main__":
    """
    img_cut normal_dir and tumor_dir
    """
    normal_dir = './low-high grade/low grade/'
    tumor_dir = './low-high grade/high grade/'
    check_dir = './imagetest/'
    img_h = 300
    img_w = 300
    threshold = 0
    # '''
    # filepath = normal_dir
    # pick_img(filepath)

    # filepath = tumor_dir
    # pick_img(filepath)
    # '''
    im = ImageDeal(height=img_h, weight=img_w, bk_cut=threshold)
    # filepath = check_dir
    pick_img(normal_dir)
    pick_img(tumor_dir)
