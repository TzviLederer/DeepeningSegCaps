"""
This script converts masks of LITS dataset to masks of lungs only
"""
import glob
import os
import matplotlib.pyplot as plt
import cv2

src_path = '/media/tzvi/1T/LITS/masks_org'
dst_path = '/media/tzvi/1T/LITS/masks'


def convert_masks(src_path, dst_path):
    # create dst directory
    try:
        os.mkdir(dst_path)
    except:
        print('directory exists')

    masks_path = glob.glob(os.path.join(src_path, '*'))
    for mask_path in masks_path:
        # read image
        # print(mask_path)
        imageRGB = cv2.imread(mask_path)
        if imageRGB is None:
            print('problem with', mask_path)
            continue
        image = imageRGB[:, :, 0]

        # # display before
        # plt.imshow(image)
        # plt.pause(0.1)

        # change color
        image[image != 0] = 255

        # # display after
        # plt.imshow(image)
        # plt.pause(0.1)

        # save
        image_name = os.path.split(mask_path)[-1]
        cv2.imwrite(os.path.join(dst_path, image_name), image)


convert_masks(src_path, dst_path)
