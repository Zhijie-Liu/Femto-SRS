# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 19:55:36 2018

@author: 1
"""

from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import core_lzj


def train(net, train_data, valid_data, num_epochs, optimizer, criterion,
          opt, cuda=0, save=True, scheduler=None, num_class=9, temp_name=None, net_path=None):
    model_name = net.__class__.__name__
    time_all = core_lzj.get_time()
    epoch_str_valid = 'Epoch {}. Train Loss: {:.6f}, Train Acc: {:.6f}, Valid Loss: {:.6f}, Valid Acc: {:.6f}, '
    epoch_str_notvalid = 'Epoch {}. Train Loss: {:.6f}, Train Acc: {:.6f}, '
    time_str_pre = 'Time {:02d}:{:02d}:{:02d}'

    device, init_flag = core_lzj.cuda_init(cuda)
    if torch.cuda.is_available():
        net.to(device)
        print('GPU is ok')

    if not net_path:
        zloss = []
        zacc = []
        xloss = []
        xacc = []
        continue_epoch = 0
        save_path = './date' + time_all + temp_name + '/'
        file = save_path + model_name + '_' + opt + 'epochs' + str(num_epochs) + '.dat'
    else:
        model_point = torch.load(net_path, map_location='cuda:' + cuda.__str__())
        continue_epoch = model_point['epoch']
        print('current epoch is', continue_epoch)
        net.load_state_dict(model_point['model'])
        print('net load finished')
        optimizer.load_state_dict(model_point['optimizer'])
        print('optimizer load finished')
        scheduler.load_state_dict(model_point['scheduler'])
        print('scheduler load finished')

        save_path = os.path.dirname(net_path) + '/'
        file = save_path + model_name + '_' + opt + 'epochs' + str(num_epochs) + time_all + '_continue.dat'
        netdata = pd.read_csv(save_path + model_name + 'params_' + opt + '.csv', index_col=0, header=0).to_numpy().T

        zloss = netdata[0, 0:continue_epoch].tolist()
        zacc = netdata[1, 0:continue_epoch].tolist()
        xloss = netdata[2, 0:continue_epoch].tolist()
        xacc = netdata[3, 0:continue_epoch].tolist()

    # save_path = './date/'
    core_lzj.check_folder_existence(save_path)
    # name = model_name + opt + '_' + str(num_epochs)
    prev_time = datetime.now()

    f = open(file, 'w')

    for epoch in range(num_epochs - continue_epoch):
        train_loss = 0
        train_acc = 0
        valid_loss = 0
        valid_acc = 0
        train_step = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im, label = im.to(device), label.to(device)
            output = net(im)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_step += 1
            train_loss += loss.item()
            train_acc += core_lzj.get_acc(output, label)
            epoch_step = epoch_str_notvalid.format(epoch + 1 + continue_epoch,
                                                   train_loss / train_step,
                                                   train_acc / train_step)
            print(epoch_step, end='\r')


        zloss.append(train_loss / len(train_data))
        zacc.append(train_acc / len(train_data))

        if (epoch + 1 + continue_epoch) % 50 == 0 and save:
            name = model_name + 'params_' + opt + 'epochs' + str(epoch + 1 + continue_epoch) + '.pkl'
            path = save_path + name
            torch.save({
                'epoch': epoch + 1 + continue_epoch,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, path)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = time_str_pre.format(h, m, s)

        if valid_data is not None:
            net = net.eval()
            with torch.no_grad():
                for im, label in valid_data:
                    if torch.cuda.is_available():
                        im, label = im.to(device), label.to(device)
                    output = net(im)
                    loss = criterion(output, label)
                    valid_loss += loss.data.item()
                    valid_acc += core_lzj.get_acc(output, label)
            xloss.append(valid_loss / len(valid_data))
            xacc.append(valid_acc / len(valid_data))
            # epoch_str = (
            #         "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, " %
            #         (epoch, train_loss / len(train_data),
            #          train_acc / len(train_data), valid_loss / len(valid_data),
            #          valid_acc / len(valid_data)))
            epoch_str = epoch_str_valid.format(epoch + 1 + continue_epoch,
                                               train_loss / len(train_data),
                                               train_acc / len(train_data),
                                               valid_loss / len(valid_data),
                                               valid_acc / len(valid_data))
        else:
            # epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
            #              (epoch, train_loss / len(train_data),
            #               train_acc / len(train_data)))
            epoch_str = epoch_str_notvalid.format(epoch + 1 + continue_epoch,
                                                  train_loss / len(train_data),
                                                  train_acc / len(train_data))

        prev_time = cur_time
        print(epoch_str + time_str)
        print(epoch_str + time_str, file=f, flush=True)
        plotzloss = np.array(zloss)
        plotzacc = np.array(zacc)
        plotxloss = np.array(xloss)
        plotxacc = np.array(xacc)

        plot_all = np.vstack((plotzloss, plotzacc, plotxloss, plotxacc))
        plotdata = pd.DataFrame(data=plot_all.T, columns=['train loss', 'train acc', 'valid loss', 'valid acc'])
        plotdata.to_csv(save_path + model_name + 'params_' + opt + '.csv')

    f.close()
    core_lzj.cuda_empty_cache(init_flag)
