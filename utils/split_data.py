"""
The script split data to subdirectories
"""
import glob
import os
import numpy as np
import shutil


path = r'/media/tzvi/1T/LITS/imgs'
dst = r'/media/tzvi/1T/LITS/imgs_split'
partition_num = 10

# load image list
image_list = glob.glob(os.path.join(path, '*'))
print(image_list)

# create destination directory
try:
    os.mkdir(dst)
except:
    print(dst, 'exists')

# create new list
split_size = int(np.ceil(len(image_list)/partition_num))
expanded_list_size = split_size*partition_num
for i in range(expanded_list_size-len(image_list)):
    image_list.append(None)

split_list = [image_list[i*split_size: (i+1)*split_size] for i in range(partition_num)]
for i, l in enumerate(split_list):
    dst_i = os.path.join(dst, str(i))
    print(dst_i)
    try:
        os.mkdir(dst_i)
    except:
        print(f'directory {i} exists')
    for image_name in l:
        shutil.copy(image_name, dst_i)
