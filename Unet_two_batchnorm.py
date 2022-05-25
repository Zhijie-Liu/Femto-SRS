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
    # base_name = 'hela'
    base_name = 'hela1228'
    fs = base_name + '/fs'
    lipids = base_name + '/lipids'
    protein = base_name + '/protein'

    dataset = core_lzj.UnetDataset(fs_dir=fs, lipids_dir=lipids, protein_dir=protein, transform=transform)
    print('dataset length is', dataset.__len__())
    # a = dataset[0]
    # im1, im2, im3 = dataset[1]
    # plt.imshow(im1.numpy()[0], cmap='gray')
    # plt.show()
    train_dataset = Subset(dataset, list(range(0, 40)))
    valid_dataset = Subset(dataset, list(range(40, 50)))
    print('train_dataset length is', train_dataset.__len__())
    print('valid_dataset length is', valid_dataset.__len__())
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=8)
    print("train_batch is", len(train_loader))
    print("test_batch is", len(valid_loader))

    model1 = UNet(n_channels=1, n_classes=1)
    model2 = UNet(n_channels=1, n_classes=1)
    params = list(model1.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print('total nubmer of trainable parameters:', nparams)

    lr = 0.001
    optimizer1 = torch.optim.RMSprop(model1.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer2 = torch.optim.RMSprop(model2.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    opt = 'RMSprop'

    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=2)

    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    # if model.n_classes > 1:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    #     criterion = nn.BCEWithLogitsLoss()

    epochs = 2800
    save_initial_epoch = 8

    epoch_str_valid = 'Epoch {}. Train Loss1: {:.6f}, Train Loss2: {:.6f}, Valid Loss1: {:.6f}, Valid Loss2: {:.6f}'
    epoch_str_notvalid = 'Epoch {}. Train Loss1: {:.6f}, Train Loss2: {:.6f}'
    epoch_str_train = 'Epoch {}. Train Loss1: {:.6f}, Train Loss2: {:.6f}, step {}/{}'
    epoch_str_trained = 'Epoch {}. Train Loss: {:.6f}, Train Loss2: {:.6f}, Valid Loss1: {:.6f}, Valid Loss2: {:.6f}, step {}/{}'
    time_str_pre = 'Time {:02d}:{:02d}:{:02d}'

    time_all = core_lzj.get_time()
    model_name = 'U-net-two-easy_batchnorm'
    t1loss, t2loss = [], []
    v1loss, v2loss = [], []
    save_path = './time_' + time_all + '_' + model_name + '/'
    file = save_path + model_name + '_' + opt + '_epochs_' + str(epochs) + '.dat'
    core_lzj.check_folder_existence(save_path)
    f = open(file, 'w')

    device, init_flag = core_lzj.cuda_init(gpu)
    if torch.cuda.is_available():
        model1.to(device)
        model2.to(device)
        print('GPU is ok')

    print('start to train the model:')

    for epoch in range(epochs):
        train_loss1, train_loss2 = 0, 0

        valid_loss1, valid_loss2 = 0, 0
        train_step = 0
        valid_step = 0
        prev_time = datetime.now()
        model1.train()
        model2.train()
        for input1, output1, output2 in train_loader:
            if torch.cuda.is_available():
                input1 = input1.to(device, dtype=torch.float32)
                output1 = output1.to(device, dtype=torch.float32)
                output2 = output2.to(device, dtype=torch.float32)
            # output_true = torch.cat([output1, output2], dim=1)
            output_net1,  output_net2= model1(input1), model2(input1)
            loss1, loss2 = criterion(output_net1, output1), criterion(output_net2, output2)
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss1.backward()
            loss2.backward()
            optimizer1.step()
            optimizer2.step()

            train_step += 1
            epoch_str = epoch_str_train.format(epoch + 1,
                                               train_loss1 / train_step,
                                               train_loss2 / train_step,
                                               train_step, len(train_loader))
            print(epoch_str, end='\r')

        t1loss.append(train_loss1 / len(train_loader))
        t2loss.append(train_loss2 / len(train_loader))

        if (epoch + 1) == int(save_initial_epoch):
            save_initial_epoch *= 1.2
            name = model_name + 'params_' + opt + '_epochs_' + str(epoch + 1) + '.pkl'
            path = save_path + name
            torch.save({
                'epoch': epoch + 1,
                'model1': model1.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'model2': model2.state_dict(),
                'optimizer2': optimizer2.state_dict()
            }, path)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = time_str_pre.format(h, m, s)

        if valid_loader is not None:
            model1.eval()
            model2.eval()
            with torch.no_grad():
                for input1, output1, output2 in valid_loader:
                    if torch.cuda.is_available():
                        input1 = input1.to(device, dtype=torch.float32)
                        output1 = output1.to(device, dtype=torch.float32)
                        output2 = output2.to(device, dtype=torch.float32)
                    output_net1, output_net2 = model1(input1), model2(input1)
                    loss1, loss2 = criterion(output_net1, output1), criterion(output_net2, output2)
                    valid_loss1 += loss1.item()
                    valid_loss2 += loss2.item()

                    valid_step += 1
                    epoch_str = epoch_str_trained.format(epoch + 1,
                                                         train_loss1 / len(train_loader),
                                                         train_loss2 / len(train_loader),
                                                         valid_loss1 / valid_step,
                                                         valid_loss2 / valid_step,
                                                         valid_step, len(valid_loader))
                    print(epoch_str, end='\r')

            v1loss.append(valid_loss1 / len(valid_loader))
            v2loss.append(valid_loss2 / len(valid_loader))

            epoch_str = epoch_str_valid.format(epoch + 1,
                                               train_loss1 / len(train_loader),
                                               train_loss2 / len(train_loader),
                                               valid_loss1 / len(valid_loader),
                                               valid_loss2 / len(valid_loader))
        else:
            # epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
            #              (epoch, train_loss / len(train_data),
            #               train_acc / len(train_data)))
            epoch_str = epoch_str_notvalid.format(epoch + 1,
                                                  train_loss1 / len(train_loader),
                                                  train_loss2 / len(train_loader))

        print(epoch_str + time_str)
        print(epoch_str + time_str, file=f, flush=True)

        plott1loss, plott2loss = np.array(t1loss), np.array(t2loss)
        plotv1loss, plotv2loss = np.array(v1loss), np.array(v2loss)

        plot_all = np.vstack((plott1loss, plott2loss, plotv1loss, plotv2loss))
        plotdata = pd.DataFrame(data=plot_all.T, columns=['train loss1', 'train loss2', 'valid loss1','valid loss2'])
        plotdata.to_csv(save_path + model_name + opt + '.csv')

    f.close()
    core_lzj.cuda_empty_cache(init_flag)




