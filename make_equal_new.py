import core_lzj
import os
import random
import shutil

path = './crossvalid low-high grade fixed/'
class_num = 3
ave_num = 5
trash_path = path + 'trash/'
core_lzj.check_folder_existence(trash_path)


def make_it_euqal(path, delet_num):
    file = core_lzj.each_img(path)
    if not os.path.exists(trash_path + path.split('/')[-1]):
        os.mkdir(trash_path + path.split('/')[-1])
    else:
        shutil.rmtree(trash_path + path.split('/')[-1])
        os.mkdir(trash_path + path.split('/')[-1])
    slice_list = random.sample(file, delet_num)
    for sli in slice_list:
        shutil.move(str(sli), trash_path + path.split('/')[-1])


def compute_class_array(list, num, average):
    temp = []
    for i in range(num):
        sample_num = len(os.listdir(list[i*average]))
        temp.append(sample_num)
    index = sorted(enumerate(temp), key=lambda x: x[1])
    return index


if __name__ == '__main__':
    ave_list = sorted(core_lzj.each_dir(path))
    array = compute_class_array(list=ave_list, num=class_num, average=ave_num)
    for idx, num in array[1:]:
        for i in range(ave_num):
            make_it_euqal(path=ave_list[i+idx*ave_num], delet_num=num-array[0][1])
