import os
import sys
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from datetime import datetime
from torch.utils.data import Dataset


number_dict = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
               5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}


# 自定义图像数据集
class MyDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.mask_img = os.listdir(self.root)
        for file in self.mask_img:
            if not file.split('.')[-1] == 'png':
                self.mask_img.remove(file)

    def __len__(self):
        return len(self.mask_img)

    def __getitem__(self, index):
        image_index = self.mask_img[index]
        img_path = os.path.join(self.root, image_index)
        img = Image.open(img_path)
        label = img_path.replace('\\', '/').split('/')[-3]
        label = eval(label)
        if self.transform:
            img = self.transform(img)
        return img, label


class MyTestDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.mask_img = os.listdir(self.root)
        for file in self.mask_img:
            if not file.split('.')[-1] == 'png':
                self.mask_img.remove(file)

    def __len__(self):
        return len(self.mask_img)

    def __getitem__(self, index):
        image_index = self.mask_img[index]
        img_path = os.path.join(self.root, image_index)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img


# 获取时间字符串
def get_time():
    time_now = datetime.now()
    time_year = time_now.year
    time_month = time_now.month
    time_day = time_now.day
    time_hour = time_now.hour
    time_minute = time_now.minute
    time_second = time_now.second
    time_all = (
            time_year.__str__().zfill(4) +
            time_month.__str__().zfill(2) +
            time_day.__str__().zfill(2) +
            time_hour.__str__().zfill(2) +
            time_minute.__str__().zfill(2) +
            time_second.__str__().zfill(2))
    return time_all


# 选择并初始化GPU
def cuda_init(gpu):
    if torch.cuda.is_available():
        # torch.cuda.set_device(gpu)
        # torch.cuda.empty_cache()
        device = torch.device('cuda:' + gpu.__str__())
        init = True
    else:
        device = torch.device('cpu')
        init = False
    return device, init


# 清空GPU
def cuda_empty_cache(init):
    if init:
        torch.cuda.empty_cache()


# 获取数字对应英文
def number_to_word(number):
    word = number_dict[number]
    return word


# 获取正确率
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


# 检查路径若不存在则创建
def check_folder_existence(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 获取子文件夹和图片列表
def eachfile(filepath):
    dir_list = []
    img_list = []
    path_dir = os.listdir(filepath)
    for all_dir in path_dir:
        if os.path.isdir(os.path.join(filepath, all_dir)):
            child = os.path.join(filepath, all_dir)
            dir_list.append(child)
        else:
            child = os.path.join(filepath, all_dir)
            img_list.append(child)
    return dir_list, img_list


# 获取图片列表
def each_img(filepath):
    img_list = []
    path_dir = os.listdir(filepath)
    for all_dir in path_dir:
        if not os.path.isdir(os.path.join(filepath, all_dir)):
            child = os.path.join(filepath, all_dir)
            img_list.append(child)
    return img_list


# 获取子文件夹列表
def each_dir(filepath):
    dir_list = []
    path_dir = os.listdir(filepath)
    for all_dir in path_dir:
        if os.path.isdir(os.path.join(filepath, all_dir)):
            child = os.path.join(filepath, all_dir)
            dir_list.append(child)
    return dir_list


# 获取单个文件路径
def get_file():
    dialog = tk.Tk()
    dialog.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("all", "*.*")], initialdir='./')
    if file_path == '':
        sys.exit(666)
    return file_path


# 获取多个文件路径
def get_files():
    dialog = tk.Tk()
    dialog.withdraw()
    files_path = filedialog.askopenfilenames(filetypes=[("all", "*.*")], initialdir='./')
    if files_path == '':
        sys.exit(6666)
    return files_path


# 获取文件夹路径
def get_directory():
    dialog = tk.Tk()
    dialog.withdraw()
    directory_dir = filedialog.askdirectory(initialdir='./')
    if directory_dir == '':
        sys.exit(66666)
    return directory_dir


# 退出程序
def exit_program():
    sys.exit(1231)


# 获取子文件路径和名字
def get_sub_directory(path_dir):
    file_list = []
    file_name = []
    for all_Dir in path_dir:
        if os.path.isdir(all_Dir):
            file_list.append(all_Dir)
            file_name.append(os.path.basename(all_Dir))
    return file_list, file_name


