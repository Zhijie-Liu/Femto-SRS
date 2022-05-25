import os
import sys
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from datetime import datetime
from torch.utils.data import Dataset
import re


def sort_key(s):
    # 排序关键字匹配
    # 匹配开头数字序号
    if s:
        try:
            c = re.findall('\d+', s)[0]
        except:
            c = -1
        return int(c)


def strsort(alist):
    alist.sort(key=sort_key)
    return alist


number_dict = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
               5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}


def get_datasets_ready(data_path):
    final_path = []
    path_list = get_datasets(data_path, final_path)
    return path_list


def get_datasets(data_path, final_path):
    sub_path = os.listdir(data_path)
    for path in sub_path:
        path_temp = os.path.join(data_path, path)
        if os.path.isdir(os.path.join(path_temp, os.listdir(path_temp)[0])):
            get_datasets(path_temp, final_path)
        else:
            final_path.append(path_temp)

    return final_path


class UnetDatasetGastric(Dataset):
    def __init__(self, fs_dir, lipids_dir, protein_dir, transform):
        self.fs_dir = fs_dir
        self.lipids_dir = lipids_dir
        self.protein_dir = protein_dir
        self.transform = transform
        # self.fs_img_dir = strsort(os.listdir(self.fs_dir))
        # self.lipids_img_dir = strsort(os.listdir(self.lipids_dir))
        # self.protein_img_dir = strsort(os.listdir(self.protein_dir))
        self.fs_img_dir = os.listdir(self.fs_dir)
        self.fs_img_dir.sort()
        # a = strsort(self.fs_img_dir)
        self.lipids_img_dir = os.listdir(self.lipids_dir)
        self.lipids_img_dir.sort()
        self.protein_img_dir = os.listdir(self.protein_dir)
        self.protein_img_dir.sort()
        # self.fs_img_dir_sort = self.fs_img_dir
        # self.fs_img_dir_sort.sort()
        # self.lipids_img_dir_sort = self.lipids_img_dir
        # self.lipids_img_dir_sort.sort()
        # self.protein_img_dir_sort = self.protein_img_dir
        # self.protein_img_dir_sort.sort()

    def __len__(self):
        return len(self.fs_img_dir)

    def __getitem__(self, index):
        fs_index = self.fs_img_dir[index]
        lipids_index = self.lipids_img_dir[index]
        protein_index = self.protein_img_dir[index]
        fs_path = os.path.join(self.fs_dir, fs_index, 's_C002.tif')
        lipids_path = os.path.join(self.lipids_dir, lipids_index, 's_C002.tif')
        protein_path = os.path.join(self.protein_dir, protein_index, 's_C002.tif')
        fs_img = Image.open(fs_path)
        lipids_img = Image.open(lipids_path)
        protein_img = Image.open(protein_path)
        # fs_img = fs_img / 4096
        # lipids_img = lipids_img / 4096
        # protein_img = protein_img / 4096
        # img = Image.open(img_path)
        # label = img_path.replace('\\', '/').split('/')[-3]
        # label = eval(label)
        if self.transform:
            fs_img = self.transform(fs_img)
            lipids_img = self.transform(lipids_img)
            protein_img = self.transform(protein_img)

        return fs_img, lipids_img, protein_img


class UnetTestGastric(Dataset):
    def __init__(self, fs_dir, transform):
        self.fs_dir = fs_dir
        self.transform = transform
        self.fs_img_dir = os.listdir(self.fs_dir)
        self.fs_img_dir.sort()
        # self.fs_img_dir_sort = self.fs_img_dir
        # self.fs_img_dir_sort.sort()
    def __len__(self):
        return len(self.fs_img_dir)

    def __getitem__(self, index):
        fs_index = self.fs_img_dir[index]
        fs_path = os.path.join(self.fs_dir, fs_index, 's_C002.tif')
        fs_img = Image.open(fs_path)
        # fs_img = fs_img / 4096
        # lipids_img = lipids_img / 4096
        # protein_img = protein_img / 4096
        # img = Image.open(img_path)
        # label = img_path.replace('\\', '/').split('/')[-3]
        # label = eval(label)
        if self.transform:
            fs_img = self.transform(fs_img)
        return fs_img, fs_index.split(' ')[0] + fs_index.split(' ')[3]


class UnetDataset(Dataset):
    def __init__(self, fs_dir, lipids_dir, protein_dir, transform):
        self.fs_dir = fs_dir
        self.lipids_dir = lipids_dir
        self.protein_dir = protein_dir
        self.transform = transform
        # self.fs_img_dir = strsort(os.listdir(self.fs_dir))
        # self.lipids_img_dir = strsort(os.listdir(self.lipids_dir))
        # self.protein_img_dir = strsort(os.listdir(self.protein_dir))
        self.fs_img_dir = os.listdir(self.fs_dir)
        # a = strsort(self.fs_img_dir)
        self.lipids_img_dir = os.listdir(self.lipids_dir)
        self.protein_img_dir = os.listdir(self.protein_dir)
        self.fs_img_dir.sort()
        self.lipids_img_dir.sort()
        self.protein_img_dir.sort()
        # self.fs_img_dir_sort = self.fs_img_dir
        # self.fs_img_dir_sort.sort()
        # self.lipids_img_dir_sort = self.lipids_img_dir
        # self.lipids_img_dir_sort.sort()
        # self.protein_img_dir_sort = self.protein_img_dir
        # self.protein_img_dir_sort.sort()

    def __len__(self):
        return len(self.fs_img_dir)

    def __getitem__(self, index):
        fs_index = self.fs_img_dir[index]
        lipids_index = self.lipids_img_dir[index]
        protein_index = self.protein_img_dir[index]
        fs_path = os.path.join(self.fs_dir, fs_index, 's_C001.tif')
        lipids_path = os.path.join(self.lipids_dir, lipids_index, 's_C001.tif')
        protein_path = os.path.join(self.protein_dir, protein_index, 's_C001.tif')
        fs_img = Image.open(fs_path)
        lipids_img = Image.open(lipids_path)
        protein_img = Image.open(protein_path)
        # fs_img = fs_img / 4096
        # lipids_img = lipids_img / 4096
        # protein_img = protein_img / 4096
        # img = Image.open(img_path)
        # label = img_path.replace('\\', '/').split('/')[-3]
        # label = eval(label)
        if self.transform:
            fs_img = self.transform(fs_img)
            lipids_img = self.transform(lipids_img)
            protein_img = self.transform(protein_img)

        return fs_img, lipids_img, protein_img


class UnetTestset(Dataset):
    def __init__(self, fs_dir, transform):
        self.fs_dir = fs_dir
        self.transform = transform
        self.fs_img_dir = os.listdir(self.fs_dir)
        self.fs_img_dir.sort()
        # self.fs_img_dir_sort = self.fs_img_dir
        # self.fs_img_dir_sort.sort()
    def __len__(self):
        return len(self.fs_img_dir)

    def __getitem__(self, index):
        fs_index = self.fs_img_dir[index]
        fs_path = os.path.join(self.fs_dir, fs_index, 's_C001.tif')
        fs_img = Image.open(fs_path)
        # fs_img = fs_img / 4096
        # lipids_img = lipids_img / 4096
        # protein_img = protein_img / 4096
        # img = Image.open(img_path)
        # label = img_path.replace('\\', '/').split('/')[-3]
        # label = eval(label)
        if self.transform:
            fs_img = self.transform(fs_img)

        return fs_img, fs_index.split('.')[0]

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


