
from torch.nn import functional as F
import pretrainedmodels.models as mymodels
from torchvision import transforms
from PIL import Image
import pandas as pd
import core_lzj
import numpy as np
import torch
import cv2
import os

if __name__ == '__main__':
    pd_0 = pd.read_csv('united/194w_probability_adjusted_matrix_0.csv', header=None)
    pd_1 = pd.read_csv('united/194w_probability_adjusted_matrix_1.csv', header=None)
    pd_2 = pd.read_csv('united/194w_probability_adjusted_matrix_2.csv', header=None)
    pd_3 = pd.read_csv('united/194w_probability_adjusted_matrix_3.csv', header=None)
    pd_no = pd.read_csv('united/194w_probability_adjusted_matrix_no.csv', header=None)

    np_data256_0 = np.array(pd_0).astype(np.uint8)
    np_data256_1 = np.array(pd_1).astype(np.uint8)
    np_data256_2 = np.array(pd_2).astype(np.uint8)
    np_data256_3 = np.array(pd_3).astype(np.uint8)
    np_data256_no = np.array(pd_no).astype(np.uint8)

    pd_lut_0 = pd.read_csv('united/lut_green.csv', header=None)
    pd_lut_1 = pd.read_csv('united/lut_pink.csv', header=None)
    pd_lut_2 = pd.read_csv('united/lut_blue.csv', header=None)
    pd_lut_3 = pd.read_csv('united/lut_red.csv', header=None)
    pd_lut_no = pd.read_csv('united/lut_no.csv', header=None)

    np_lut0 = np.array(pd_lut_0).astype(np.uint8)
    np_lut1 = np.array(pd_lut_1).astype(np.uint8)
    np_lut2 = np.array(pd_lut_2).astype(np.uint8)
    np_lut3 = np.array(pd_lut_3).astype(np.uint8)
    np_lutno = np.array(pd_lut_no).astype(np.uint8)

    np_cv2_lut0 = np.flip(np_lut0, 1)
    np_cv2_lut1 = np.flip(np_lut1, 1)
    np_cv2_lut2 = np.flip(np_lut2, 1)
    np_cv2_lut3 = np.flip(np_lut3, 1)
    np_cv2_lutno = np.flip(np_lutno, 1)

    lut0 = np.expand_dims(np_cv2_lut0, axis=0)
    lut1 = np.expand_dims(np_cv2_lut1, axis=0)
    lut2 = np.expand_dims(np_cv2_lut2, axis=0)
    lut3 = np.expand_dims(np_cv2_lut3, axis=0)
    lutno = np.expand_dims(np_cv2_lutno, axis=0)

    prob_0 = cv2.LUT(np.dstack([np_data256_0] * 3), lut0)
    prob_1 = cv2.LUT(np.dstack([np_data256_1] * 3), lut1)
    prob_2 = cv2.LUT(np.dstack([np_data256_2] * 3), lut2)
    prob_3 = cv2.LUT(np.dstack([np_data256_3] * 3), lut3)
    prob_no = cv2.LUT(np.dstack([np_data256_no] * 3), lutno)

    dir_path = 'united'
    file_name = '194w'

    cv2.imwrite(os.path.join(dir_path, file_name) + 'lut_0.png', prob_0)
    cv2.imwrite(os.path.join(dir_path, file_name) + 'lut_1.png', prob_1)
    cv2.imwrite(os.path.join(dir_path, file_name) + 'lut_2.png', prob_2)
    cv2.imwrite(os.path.join(dir_path, file_name) + 'lut_3.png', prob_3)
    cv2.imwrite(os.path.join(dir_path, file_name) + 'lut_no.png', prob_no)
    cv2.imwrite(os.path.join(dir_path, file_name) + 'lut.png', prob_0 + prob_2 + prob_3 + prob_no)
