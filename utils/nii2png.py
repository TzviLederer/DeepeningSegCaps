"""
The script convert nii files to png
"""
import os
import glob

# destination
images_path = '/media/tzvi/1T/LITS/data'
masks_path = '/media/tzvi/1T/LITS/masks'

# get images and masks to sort
src_path = '/media/tzvi/1T/LITS/TrainingBatch2'
seg_files = glob.glob(os.path.join(src_path, 'seg*.nii'))
image_files = glob.glob(os.path.join(src_path, 'vol*.nii'))

seg_files.sort()
image_files.sort()

for seg_file_i, image_file_i in zip(seg_files, image_files):
    num_seg = seg_file_i.split('-')[-1].split('.')[0]
    num_image = image_file_i.split('-')[-1].split('.')[0]
    assert num_image == num_seg, 'not same file'

    os.system(f'med2image -i {image_file_i} -d {images_path} -o image-{num_image}.png -s -1')
    os.system(f'med2image -i {seg_file_i} -d {masks_path} -o image-{num_seg}.png -s -1')
    print(seg_file_i, image_file_i)
