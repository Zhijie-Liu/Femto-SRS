
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


num_class = 2
gpu = 3
img_size = 50
net_img_size = 300
thershold = 0.2
thershold2 = 0
step = int(net_img_size/img_size)

preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# preprocess1 = transforms.Compose([
#     # transforms.ToTensor(),
#     transforms.Resize((299, 299))
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# preprocess2 = transforms.Compose([
#     # transforms.Resize((299, 299)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# def get_one(img, h_location=0, w_location=0):
#     # img = Image.new('RGB', (net_img_size, net_img_size))
#     for i in range(0, step):
#         for j in range(0, step):
#             x1, y1 = (w_location + j)*50, (h_location + i)*50
#             x2, y2 = (w_location + j)*50 + 300, (h_location + i)*50 + 300
#             img_temp = img.crop(j*50, i*50, j*50 + 300, i*50 + 300)
#             # img_path = path + '/' + basename + '_' + str(h_location + i + 1) + '_' + str(w_location + j + 1) + '.png'
#             # img.paste(Image.open(img_path), (j*50, i*50))
#     return preprocess(img_temp).unsqueeze(0)


def get_probability_matrix(net, net2, cuda, nrow, ncol, img_mosaic):
    matrix_0, matrix_1, matrix_2, matrix_3, matrix_no = np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol))
    for i in range(0, nrow - step + 1):
        for j in range(0, ncol - step + 1):
            img_temp = img_mosaic[i * img_size: i * img_size + net_img_size, j * img_size: j * img_size + net_img_size]
            img = preprocess(Image.fromarray(cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB))).unsqueeze(0)
            if torch.cuda.is_available():
                img = img.to(cuda)
            output = net(img)
            output_prob = F.softmax(output, dim=1).cpu().detach().squeeze().numpy()
            print(output_prob)
            # _, pred_label = output.max(1)
            # temp_matrix = np.zeros((6, 6))
            if output_prob[0] - output_prob[1] > thershold:
                matrix_0[i:i + step, j:j + step] += 1
            elif output_prob[0] - output_prob[1] <= -thershold:
                matrix_1[i:i + step, j:j + step] += 1
                output2 = net2(img)
                output_prob2 = F.softmax(output2, dim=1).cpu().detach().squeeze().numpy()
                print(output_prob2)
                if output_prob2[0] - output_prob2[1] > thershold2:
                    matrix_2[i:i + step, j:j + step] += 1
                else:
                    matrix_3[i:i + step, j:j + step] += 1
            else:
                matrix_no[i:i + step, j:j + step] += 1

            print(i, j)

    return matrix_0, matrix_1, matrix_2, matrix_3, matrix_no


if __name__ == '__main__':
    select = 0
    if select == 0:
        img_dir = 'united/194w.tif'
        net_path = 'united/normal to tumor net.pkl'
        net2_path = 'united/low to high net.pkl'
    # elif select == 1:
    #     img_dir = core_lzj.get_file()
    #     net_path = core_lzj.get_file()
    # elif select == 2:
    #     img_dir = core_lzj.get_file()
    #     net_path = 'date20200905213524crossvalid1 two classclass/InceptionResNetV2params_Adamepochs600.pkl'
    # elif select == 3:
    #     img_dir = 'check/027h-2_18_21'
    #     net_path = core_lzj.get_file()
    else:
        img_dir = []
        net_path = []
        net2_path = []
        core_lzj.exit_program()

    img_raw = cv2.imread(img_dir)
    img_nrow, img_ncol = int(img_raw.shape[0] / img_size), int(img_raw.shape[1] / img_size)
    # img_nrow, img_ncol = int(img_dir.split('_')[-2]), int(img_dir.split('_')[-1])
    device, init_flag = core_lzj.cuda_init(gpu)
    # img_basename = img_dir.split('/')[-1]
    models = mymodels.inceptionresnetv2(num_classes=num_class, pretrained=False)
    models2 = mymodels.inceptionresnetv2(num_classes=num_class, pretrained=False)

    if torch.cuda.is_available():
        models.to(device)
        models2.to(device)

    # img = get_one(h_location=0, w_location=0, path='probability/194w_99_108', basename='194w_99_108')
    models.load_state_dict(torch.load(net_path, map_location='cuda:' + gpu.__str__()))
    models2.load_state_dict(torch.load(net2_path, map_location='cuda:' + gpu.__str__()))
    models.eval()
    models2.eval()
    prob_matrix_0, prob_matrix_1, prob_matrix_2, prob_matrix_3, prob_matrix_no = get_probability_matrix(net=models, net2=models2, cuda=device, nrow=img_nrow,
                                                                          ncol=img_ncol, img_mosaic=img_raw)
    # probability_data = pd.DataFrame(data=probability_matrix)
    # probability_data.to_csv(os.path.dirname(img_dir) + '/' + os.path.basename(img_dir) + '_' + myfunctions.get_time() +
    #                         '_probability.csv', header=False, index=False)

    dim_row = np.concatenate((np.arange(1, step + 1, 1), np.ones(img_nrow - step * 2) * step, np.arange(step, 0, -1)))
    dim_col = np.concatenate((np.arange(1, step + 1, 1), np.ones(img_ncol - step * 2) * step, np.arange(step, 0, -1)))
    adjust_matrix = dim_row.reshape(-1, 1) * dim_col.reshape(1, -1)
    adjusted_matrix_0 = np.round(prob_matrix_0 / adjust_matrix * 255).astype(np.uint8)
    adjusted_matrix_1 = np.round(prob_matrix_1 / adjust_matrix * 255).astype(np.uint8)
    adjusted_matrix_2 = np.round(prob_matrix_2 / adjust_matrix * 255).astype(np.uint8)
    adjusted_matrix_3 = np.round(prob_matrix_3 / adjust_matrix * 255).astype(np.uint8)
    adjusted_matrix_no = np.round(prob_matrix_no / adjust_matrix * 255).astype(np.uint8)
    # adjusted_data = pd.DataFrame(data=adjusted_matrix)
    # adjusted_data256 = pd.DataFrame(data=adjusted_matrix256)
    # adjusted_data.to_csv(os.path.dirname(img_dir) + '/' + os.path.basename(img_dir) + '_' + myfunctions.get_time() +
    #                      '_probability_adjusted.csv', header=False, index=False)
    # adjusted_data256.to_csv(os.path.dirname(img_dir) + '/' + os.path.basename(img_dir) + '_' + myfunctions.get_time() +
    #                         '_probability_adjusted256.csv', header=False, index=False)
    adjusted_matrix_0_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
    adjusted_matrix_1_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
    adjusted_matrix_2_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
    adjusted_matrix_3_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
    adjusted_matrix_no_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)

    for i in range(img_nrow):
        for j in range(img_ncol):
            adjusted_matrix_0_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
                adjusted_matrix_0[i, j]
            adjusted_matrix_1_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
                adjusted_matrix_1[i, j]
            adjusted_matrix_2_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
                adjusted_matrix_2[i, j]
            adjusted_matrix_3_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
                adjusted_matrix_3[i, j]
            adjusted_matrix_no_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
                adjusted_matrix_no[i, j]

    prob_0 = cv2.applyColorMap(adjusted_matrix_0_raw, cv2.COLORMAP_JET)
    # prob_0_resize = cv2.resize(prob_0, (img_raw.width, img_raw.height))

    prob_1 = cv2.applyColorMap(adjusted_matrix_1_raw, cv2.COLORMAP_JET)
    # prob_1_resize = cv2.resize(prob_1, (img_raw.width, img_raw.height))
    prob_2 = cv2.applyColorMap(adjusted_matrix_2_raw, cv2.COLORMAP_JET)

    prob_3 = cv2.applyColorMap(adjusted_matrix_3_raw, cv2.COLORMAP_JET)

    prob_no = cv2.applyColorMap(adjusted_matrix_no_raw, cv2.COLORMAP_JET)
    # prob_no_resize = cv2.resize(prob_no, (img_raw.width, img_raw.height))

    dir_path = os.path.dirname(img_dir)
    file_name = os.path.basename(img_dir).split('.')[0]

    adjusted_data256_matrix_0 = pd.DataFrame(data=adjusted_matrix_0_raw)
    adjusted_data256_matrix_0.to_csv(
        os.path.dirname(img_dir) + '/' + file_name + '_' + core_lzj.get_time() +
        '_probability_adjusted_matrix_0.csv', header=False, index=False)

    adjusted_data256_matrix_1 = pd.DataFrame(data=adjusted_matrix_1_raw)
    adjusted_data256_matrix_1.to_csv(
        os.path.dirname(img_dir) + '/' + file_name + '_' + core_lzj.get_time() +
        '_probability_adjusted_matrix_1.csv', header=False, index=False)

    adjusted_data256_matrix_2 = pd.DataFrame(data=adjusted_matrix_2_raw)
    adjusted_data256_matrix_2.to_csv(
        os.path.dirname(img_dir) + '/' + file_name + '_' + core_lzj.get_time() +
        '_probability_adjusted_matrix_2.csv', header=False, index=False)

    adjusted_data256_matrix_3 = pd.DataFrame(data=adjusted_matrix_3_raw)
    adjusted_data256_matrix_3.to_csv(
        os.path.dirname(img_dir) + '/' + file_name+ '_' + core_lzj.get_time() +
        '_probability_adjusted_matrix_3.csv', header=False, index=False)

    adjusted_data256_matrix_no = pd.DataFrame(data=adjusted_matrix_no_raw)
    adjusted_data256_matrix_no.to_csv(
        os.path.dirname(img_dir) + '/' + file_name + '_' + core_lzj.get_time() +
        '_probability_adjusted_matrix_no.csv', header=False, index=False)

    cv2.imwrite(os.path.join(dir_path, file_name) + '50_0.png', prob_0)
    cv2.imwrite(os.path.join(dir_path, file_name) + '50_1.png', prob_1)
    cv2.imwrite(os.path.join(dir_path, file_name) + '50_2.png', prob_2)
    cv2.imwrite(os.path.join(dir_path, file_name) + '50_3.png', prob_3)
    cv2.imwrite(os.path.join(dir_path, file_name) + '50_no.png', prob_no)

    np_data256_0 = np.array(adjusted_matrix_0_raw).astype(np.uint8)
    np_data256_1 = np.array(adjusted_matrix_1_raw).astype(np.uint8)
    np_data256_2 = np.array(adjusted_matrix_2_raw).astype(np.uint8)
    np_data256_3 = np.array(adjusted_matrix_3_raw).astype(np.uint8)
    np_data256_no = np.array(adjusted_matrix_no_raw).astype(np.uint8)

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

    cv2.imwrite(os.path.join(dir_path, file_name) + '_lut_0.png', prob_0)
    cv2.imwrite(os.path.join(dir_path, file_name) + '_lut_1.png', prob_1)
    cv2.imwrite(os.path.join(dir_path, file_name) + '_lut_2.png', prob_2)
    cv2.imwrite(os.path.join(dir_path, file_name) + '_lut_3.png', prob_3)
    cv2.imwrite(os.path.join(dir_path, file_name) + '_lut_no.png', prob_no)
    cv2.imwrite(os.path.join(dir_path, file_name) + '_lut.png', prob_0 + prob_2 + prob_3 + prob_no)