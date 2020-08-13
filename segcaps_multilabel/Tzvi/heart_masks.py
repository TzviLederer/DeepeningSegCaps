import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


path = r'/home/tzvi/PycharmProjects/Matwo-CapsNet/Dataset/masksMatwo'
dst_path = r'/home/tzvi/PycharmProjects/Matwo-CapsNet/Dataset/masks_heart'

try:
    os.mkdir(dst_path)
except FileExistsError:
    print('directory exists')

for im_path in glob.glob(os.path.join(path, '*')):
    mask = cv2.imread(im_path)
    heart_mask = mask[:,:,0] == 5
    plt.imshow(np.double(heart_mask))
    # plt.pause(0.01)
    cv2.imwrite(os.path.join(dst_path, os.path.split(im_path)[-1]), np.uint8(heart_mask*255))
    print(im_path)