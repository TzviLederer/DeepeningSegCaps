import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

imgs_path = r'/media/tzvi/1T/Matwo/imgs'
mask_path = r'/media/tzvi/1T/Matwo/masks'
dst_path = r'/media/tzvi/1T/Matwo/masks_multilabel'
new_shape = (128, 128)
class_num = 6

try:
    os.mkdir(dst_path)
except FileExistsError:
    print('directory exists')

for im_path in glob.glob(os.path.join(mask_path, '*')):
    mask = cv2.imread(im_path)
    mask = mask[:, :, 0]
    mask = cv2.resize(mask, new_shape)

    # create multilabel mask
    multilabel_mask = np.zeros((new_shape[0], new_shape[1], class_num))
    for i in range(class_num):
        multilabel_mask[:, :, i] = mask == i

    # read image
    image = cv2.imread(os.path.join(imgs_path, os.path.split(im_path)[-1]))
    image = cv2.resize(image, new_shape)[:, :, 0]
    image = np.expand_dims(image, axis=2)

    npz_name = os.path.split(im_path)[-1]
    npz_name = npz_name.split('.')[0]
    np.savez(os.path.join(dst_path, npz_name), img=image, mask=multilabel_mask)
    print(im_path)
print('image shape:', image.shape)
print('mask shape:', multilabel_mask.shape)