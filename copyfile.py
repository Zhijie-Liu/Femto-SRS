import core_lzj
import shutil
import math
import random


def image_copy(path, ratio):
    num_split = math.modf(ratio)
    img_list = core_lzj.eachfile(path)
    num_fig = int(len(img_list) * num_split[0])
    select_list = random.sample(img_list, num_fig)
    for img in select_list:
        temp = img.split('.')[0] + '_copy.png'.format(0)
        shutil.copyfile(img, temp)

    for idx in range(int(num_split[1])):
        for img in img_list:
            temp = img.split('.')[0] + '_copy{}.png'.format(idx + 1)
            shutil.copyfile(img, temp)

    print('copy finished')


if __name__ == '__main__':
    img_path = core_lzj.get_directory()
    ratio_num = 7
    image_copy(img_path, ratio_num)
