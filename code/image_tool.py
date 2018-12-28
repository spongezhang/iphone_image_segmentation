from __future__ import print_function
import shutil
import os
import glob
import cv2
import numpy as np

#path = '../data/img/training/'
#output_path = '../data/img/training/'
path = '../data/mask/95_masks_ori/'
output_path = '../data/mask/95_masks/'

for img_name in glob.glob(path + '/*.png'):
    pure_name = img_name.split('/')[-1]
    pure_name = pure_name.split('.')[0]
    print(pure_name)
    number = int(pure_name[-2:])
    pure_name = pure_name[:-3]
    pure_name = pure_name + '-{}'.format(number)
    im_gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    indices = np.where(im_gray > 100)
    im_gray[:,:] = 0
    im_gray[indices] = 255
    im_gray = cv2.resize(im_gray, (512, 512))
    cv2.imwrite('{}/{}.png'.format(output_path, pure_name), im_gray)
