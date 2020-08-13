import os
import glob
import shutil


image_root_dir = r'/media/tzvi/1T/LITS/imgs_split'
mask_dir = r'/media/tzvi/1T/LITS/masks'
mask_root_dir = r'/media/tzvi/1T/LITS/masks_split'

# create root directory for masks
try:
    os.mkdir(mask_root_dir)
except:
    print(f'directory {mask_root_dir} exists.')

# read image sub-directories
for image_root_dir_i in glob.glob(os.path.join(image_root_dir, '*')):
    # create subdirectory
    print(f'image dictionary: {image_root_dir_i}')
    try:
        mask_subdir = os.path.join(mask_root_dir, os.path.split(image_root_dir_i)[-1])
        os.mkdir(mask_subdir)
        print(f'{mask_subdir} was created')
    except:
        print(f'{mask_subdir} already exists')

    # list of images in the subdirectory
    image_list_i = glob.glob(os.path.join(image_root_dir_i, '*.png'))
    for image_path in image_list_i:
        image_name = os.path.split(image_path)[-1]
        mask_path = os.path.join(mask_dir, image_name)
        try:
            shutil.copy(mask_path, mask_subdir)
        except:
            print(f'{mask_path} dosnt exists')
