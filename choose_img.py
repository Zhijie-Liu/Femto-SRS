# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:09:30 2018

@author: 1
"""
import os
import random
import shutil
import core_lzj
import numpy as np

trash_path = './trash/'


class Chooseimg:
    def __init__(self, filepath):
        self.filepath = filepath

    def num_list(self, ratio):
        files = core_lzj.each_dir(self.filepath)
        num_fig = []
        for file in files:
            if os.path.isdir(file):
                file_list = os.listdir(file)
                dirname, basename = os.path.splitdrive(file)
                print('num_fig of folder', basename, ' = ', len(file_list))
                num_fig.append(int(len(file_list) * ratio))
        print('fig_list =', num_fig)
        print('')
        return num_fig

    def random_select(self, num_list, test_path):
        print(test_path)
        files = core_lzj.each_dir(self.filepath)
        dest_path = test_path

        core_lzj.check_folder_existence(dest_path)

        print('num_list =', num_list)
        len_list = len(num_list)
        print('len_list =', len_list)
        print('dest_path =', dest_path)
        inum = 0
        for file in files:
            if os.path.isdir(file):
                dirname, basename = os.path.splitdrive(file)
                file_list = os.listdir(file)
                print('num_fig of folder', basename, ' = ', len(file_list))

                if inum < len_list:
                    slice_list = random.sample(file_list, num_list[inum])
                    inum += 1
                    for sli in slice_list:
                        name = str(sli)
                        path = basename + '/' + name
                        shutil.move(path, dest_path)

        num_fig = os.listdir(dest_path)
        print('num_fig of floder ', test_path, ' = ', len(num_fig))


if __name__ == '__main__':
    check_on = True  # True for first time
    filepath = './check/0'
    # crossvalid_number = 5
    ch_img = Chooseimg(filepath=filepath)
    image_list = ch_img.num_list(ratio=0.2)
    subpath = 'check/random choose/'
    core_lzj.check_folder_existence(subpath)
    if not check_on:
        ch_img.random_select(num_list=image_list, test_path=subpath)

