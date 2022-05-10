# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:49:05 2018

@author: 1
"""
import os
import torch
import core_lzj
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import pretrainedmodels.models as mymodels
import numpy as np
from datetime import datetime
import pandas as pd
import re
# re_im_size=299  #make all im to the same size
# crop_im_size=196  #im_w=96 im_h=96

test_transform = transforms.Compose([
    # transforms.Resize(re_im_size),
    # transforms.CenterCrop(crop_im_size),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# check_dir = './check/'
# check_dir = './three grade test/1/'
batch_size = 1
num_epochs = 1
num_class = 3
validname = 'epoch 60 inception renset v2 crossvalid1 diff'
gpu = 1


def get_imgdata(file_dir):
    this_folder = ImageFolder(root=file_dir + '/', transform=test_transform)
    this_data = DataLoader(this_folder, batch_size=batch_size, shuffle=False)
    # this_name = this_folder.mask_img
    return this_data


def check(net, check_data, class_num, cuda):
    if torch.cuda.is_available():
        net.to(cuda)
        print('check is starting')
    probability_list=[]
    every_class_num = np.zeros(class_num)
    evert_class_name = []
    [evert_class_name.append([]) for _ in range(class_num)]
    fig_num = 0
    label = 999
    for im, label in check_data:
        net.eval()
        if torch.cuda.is_available():
            im, label = im.to(cuda), label.to(cuda)  # (bs, 3, h, w)
        output = net(im)
        probaility=torch.nn.functional.softmax(output,dim=1)
        probability_list.extend(probaility.cpu().detach().numpy())
        _, pred_label = output.max(1)
        every_class_num[pred_label.data[0]] += 1
        # evert_class_name[pred_label.data[0]].append(file_name[fig_num])
        fig_num += 1
    # label_flag = label
    class_judge_temp = every_class_num / fig_num
    return probability_list


def get_results(output, label):
    img_index, pred_label = output.max(1)
    # print('label ---> ',label.data[0])
    # print('pred_label = ',pred_label.data[0])
    if pred_label.data[0] == 1:
        pass
        # print ('tumor ---> ')
    else:
        pass
        # print ('normal ------>')


if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    select = False
    if select:
        check_dir = core_lzj.get_directory()
        net_path = core_lzj.get_file()
    else:
        check_dir = 'crossvalid low-high grade fixed train/0'
        net_path = 'date20210111015257crossvalid low-high grade fixed/InceptionResNetV2params_Adamepochs60.pkl'
    # path_dir = core_lzj.each_dir(filepath=check_dir)
    # img_list, img_name = core_lzj.get_sub_directory(path_dir=path_dir)
    my_model = mymodels.inceptionresnetv2(num_classes=num_class, pretrained=False)
    my_model.load_state_dict(torch.load(net_path, map_location='cuda:' + gpu.__str__())['model'])
    oneNET_judge = []
    device, init_flag = core_lzj.cuda_init(gpu)
    data = get_imgdata(file_dir=check_dir)
    probability_list = check(net=my_model, check_data=data, class_num=num_class, cuda=device)

    # everyNETdata_tumor = pd.DataFrame(data=oneNET_judge, index=img_name, columns=list(range(num_class)))
    # everyNETdata_tumor.to_csv(check_dir + validname + '_' + 'oneNET' + core_lzj.get_time() + '_acc_result.csv')
    oneNETdata_tumor = pd.DataFrame(data=probability_list)
    oneNETdata_tumor.to_csv(check_dir + '0_' + 'oneNET' + core_lzj.get_time() + '_acc_result.csv', header=False, index=False)
    core_lzj.cuda_empty_cache(init_flag)




