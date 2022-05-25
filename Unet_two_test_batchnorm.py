from datetime import datetime
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor
from U_net.unet_model_batchnorm import UNet
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import core_lzj
import numpy as np
import pandas as pd
import torch
from torch import nn
import cv2


gpu = 3
val_percent = 0.2


transform = transforms.Compose([
    # transforms.Resize(re_im_size),
    # transforms.RandomCrop(crop_im_size, padding=0),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
    transforms.ToTensor(),
    # transforms.Normalize(0.485, 0.229)
])

if __name__ == "__main__":
    base_name = 'hela1228'
    fs = base_name + '/fs'
    # lipids = base_name + '/lipids'
    # protein = base_name + '/protein'

    select = 1
    if select == 0:
        # check_dir = core_lzj.get_directory()
        net_path = core_lzj.get_file()
    else:
        # check_dir = 'check/0/145x_12_12'
        net_path = 'time_20210904181927_U-net-two-easy_batchnorm/U-net-two-easy_batchnormparams_RMSprop_epochs_2734.pkl'

    dataset = core_lzj.UnetTestset(fs_dir=fs, transform=transform)
    print('dataset length is', dataset.__len__())
    # a = dataset[0]
    # train_dataset = Subset(dataset, list(range(0, 80)))
    valid_dataset = Subset(dataset, list(range(40, 50)))
    # print('train_dataset length is', train_dataset.__len__())
    print('valid_dataset length is', valid_dataset.__len__())
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)
    # print("train_batch is", len(train_loader))
    print("valid_batch is", len(valid_loader))

    device, init_flag = core_lzj.cuda_init(gpu)

    model1, model2 = UNet(n_channels=1, n_classes=1), UNet(n_channels=1, n_classes=1)
    model1.load_state_dict(torch.load(net_path, map_location='cuda:' + gpu.__str__())['model1'])
    model2.load_state_dict(torch.load(net_path, map_location='cuda:' + gpu.__str__())['model2'])

    if torch.cuda.is_available():
        model1.to(device)
        model2.to(device)
    print('check is starting')
    model1.eval()
    model2.eval()

    save_path = '20201228 batchnorm hela net2734 testsets'
    core_lzj.check_folder_existence(save_path)

    for im, name in valid_loader:
        print(name[0])
        if torch.cuda.is_available():
            im = im.to(device, dtype=torch.float32)
        output1, output2 = model1(im), model2(im)
        path = os.path.join(save_path, name[0])
        core_lzj.check_folder_existence(path)
        img1, img2 = output1[0][0].cpu().detach().numpy(), output2[0][0].cpu().detach().numpy()
        img1_uint16, img2_uint16 = (img1 - img1.min()) / (img1.max() - img1.min()) * 65535, (img2 - img2.min()) / (img2.max() - img2.min()) * 65535
        cv2.imwrite(os.path.join(path, name[0] + '-lipids.png'), img1_uint16.astype(np.uint16))
        cv2.imwrite(os.path.join(path, name[0] + '-protein.png'), img2_uint16.astype(np.uint16))

        # img1_data = pd.DataFrame(data=img1)
        # img2_data = pd.DataFrame(data=img2)
        # img1_data.to_csv(os.path.join(path, name[0] + '-lipids.csv'), header=False, index=False)
        # img2_data.to_csv(os.path.join(path, name[0] + '-protein.csv'), header=False, index=False)
        #
        # a=1


