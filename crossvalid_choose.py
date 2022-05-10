# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:09:30 2018

@author: 1
"""
import os
import choose_img
import core_lzj

trash_path = 'trash/'

crossvalid_number = 5
class_number = 3
all_path = 'low-high grade fixed/'

if __name__ == '__main__':
    check_on = False
    for dir_num in range(class_number):
        file_path = all_path + dir_num.__str__()
        ch_img = choose_img.Chooseimg(filepath=file_path)
        image_list = ch_img.num_list(ratio=1/crossvalid_number)
        for i in range(0, crossvalid_number):
            if not check_on:
                mainpath = 'crossvalid low-high grade fixed/'
                core_lzj.check_folder_existence(mainpath)
                subpath = mainpath + dir_num.__str__() + '-' + (i + 1).__str__() + '/'
                core_lzj.check_folder_existence(subpath)
                ch_img.random_select(num_list=image_list, test_path=subpath)
