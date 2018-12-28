#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: align_image.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-15-2018
#  Last Modified: Sun Oct 14 11:27:53 2018
#
#  Usage: python align_image.py -h
#  Description:
#
#  Copyright (C) 2018 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

import cv2
import numpy as np
import argparse
import glob
import os


standard_height = 512
standard_width = 512
small_height = 256
small_width = 256

def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', default='P14_100016572_56_r_crop.jpg', help='Name of training AOI')
    parser.add_argument('--data_dir', default='../data/', help='folder to image data')
    args = parser.parse_args() 

    # Read 8-bit color image.
    # This is an image in which the three channels are
    # concatenated vertically.
    source_name = args.source.split(".")[0]
 
    im1 =  cv2.imread("{}/img/training_color/{}".format(args.data_dir, args.source), cv2.IMREAD_UNCHANGED);
    im1 = cv2.resize(im1, (standard_width,standard_height))
    im1_gray_ori = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    # Find the width and height of the color image
    im1_gray = cv2.resize(im1_gray_ori, (small_width, small_height))
    sz = im1_gray.shape
    height = sz[0]
    width = sz[1]
    im1_color_small = cv2.resize(im1, (height,width)) 
    training_mask_list = []
    for i in [3]:#range(4):
        mask1 = cv2.imread("{}/mask/training_masks/{}-{}.png".format(args.data_dir, source_name, i+1), cv2.IMREAD_GRAYSCALE);
        mask1 = cv2.resize(mask1, (standard_width,standard_height), cv2.INTER_NEAREST)
        mask1[mask1>100]=255
        mask1[mask1<=100]=0
        cv2.imwrite("{}/mask/training_masks_aligned/{}-{}.png".format(args.data_dir, source_name, i+1), mask1)
        training_mask_list.append(mask1.copy())

    ##cv2.imwrite("./data/small_img/small_img_1.jpg", im1_color_small)
    ##cv2.imwrite("./data/gradient_img/img_1.jpg", get_gradient(im1_gray))
    cv2.imwrite("{}/img/training/{}.png".format(args.data_dir, source_name), im1_gray_ori)
    #for idx, mask in enumerate(training_mask_list):
    #    cv2.imwrite("{}/mask/training_masks_aligned/{}-{}.png".format(args.data_dir, source_name, idx+1), mask1)

    for filename in glob.glob("{}/img/test_img_color/*.*".format(args.data_dir)):
        #if args.source in filename:
        #    continue
        print(filename)
        if 'png' in filename or 'jpeg' in filename or 'jpg' in filename:
            filename = filename.split('/')[-1]
            purefilename = filename.split('.')[0]
            im2 = cv2.imread("{}/img/test_img_color/{}".format(args.data_dir,filename), cv2.IMREAD_UNCHANGED)
            im2 = cv2.resize(im2, (standard_width,standard_height))
            
            test_mask_list = []
            for i in [3]:#range(4):
                print("{}/mask/all_masks/{}-{}.png".format(args.data_dir, purefilename, i+1))
                mask2 =  cv2.imread("{}/mask/all_masks/{}-{}.png".format(args.data_dir, purefilename, i+1), cv2.IMREAD_GRAYSCALE)
                mask2 = cv2.resize(mask2, (standard_width,standard_height), cv2.INTER_NEAREST)
                test_mask_list.append(mask2.copy())

            im2_gray_ori = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
            im2_gray = cv2.resize(im2_gray_ori, (height,width))
            im2_color_small = cv2.resize(im2, (height,width)) 

            # Allocate space for aligned image
            im_aligned = np.zeros((height,width,3), dtype=np.uint8)
 
            # Define motion model
            warp_mode = cv2.MOTION_AFFINE
            # Set the warp matrix to identity.
            if warp_mode == cv2.MOTION_HOMOGRAPHY :
                warp_matrix = np.eye(3, 3, dtype=np.float32)
            else :
                warp_matrix = np.eye(2, 3, dtype=np.float32)
 
            # Set the stopping criteria for the algorithm.
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-3)
            # Warp the blue and green channels to the red channel
            for j in range(0,1):
                (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
                if warp_mode == cv2.MOTION_HOMOGRAPHY :
                    # Use Perspective warp when the transformation is a Homography
                    forword_scale_matrix = np.eye(2,dtype = np.float)
                    forword_scale_matrix[0,0] = standard_width/float(width)
                    forword_scale_matrix[1,1] = standard_height/float(height)
                    inverse_scale_matrix = np.eye(3,dtype = np.float)
                    inverse_scale_matrix[0,0] = width/float(standard_width)
                    inverse_scale_matrix[1,1] = height/float(standard_height)
                    new_warp_matrix = np.matmul(np.matmul(forword_scale_matrix,warp_matrix), inverse_scale_matrix)

                    im2_aligned = cv2.warpPerspective (im2, new_warp_matrix, (standard_width,standard_height),\
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    im2_gray_aligned = cv2.warpPerspective (im2_gray, warp_matrix, (width,height),\
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

                    mask_flag = np.logical_or(np.logical_or(im2_aligned[:,:,0]<200, im2_aligned[:,:,1]<200), im2_aligned[:,:,2]<200)
                    im_merge[mask_flag,1] = im2_gray_aligned[mask_flag]
                else :
                    forword_scale_matrix = np.eye(2,dtype = np.float)
                    forword_scale_matrix[0,0] = standard_width/float(width)
                    forword_scale_matrix[1,1] = standard_height/float(height)
                    inverse_scale_matrix = np.eye(3,dtype = np.float)
                    inverse_scale_matrix[0,0] = width/float(standard_width)
                    inverse_scale_matrix[1,1] = height/float(standard_height)
                    new_warp_matrix = np.matmul(np.matmul(forword_scale_matrix,warp_matrix), inverse_scale_matrix)
                   
                    # print(filename)
                    name = os.path.splitext(filename)[0]
                    # print(name)
                    matrix_path = '../data/transform_matrix/{}/'.format(source_name)
                    try:
                        os.stat(matrix_path)
                    except:
                        os.makedirs(matrix_path)
                    np.save(matrix_path + name, new_warp_matrix)

                    # Use Affine warp when the transformation is not a Homography
                    im2_aligned = cv2.warpAffine(im2, new_warp_matrix, (standard_width,standard_height),\
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderValue = (255,255,255));
                    im2_small_aligned = cv2.warpAffine(im2_color_small, warp_matrix, (width,height),\
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderValue = (255,255,255));
                    im2_gray_aligned = cv2.warpAffine(im2_gray_ori, new_warp_matrix, (standard_width, standard_height),
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderValue = (255,255,255));

            # Show final output
                    try:
                        os.stat("{}/img/test_img_aligned/{}/".format(args.data_dir, source_name))
                    except:
                        os.makedirs("{}/img/test_img_aligned/{}/".format(args.data_dir, source_name))

                    try:
                        os.stat("{}/mask/all_masks_aligned/{}/".format(args.data_dir, source_name))
                    except:
                        os.makedirs("{}/mask/all_masks_aligned/{}/".format(args.data_dir, source_name))

                    cv2.imwrite("{}/img/test_img_aligned/{}/{}.png".format(args.data_dir, source_name, purefilename), im2_gray_aligned)

                    for idx, mask in enumerate(test_mask_list):
                        mask2_aligned = cv2.warpAffine(mask, new_warp_matrix, (standard_width,standard_height),\
                                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP, borderValue = (1))
                        mask2_aligned[mask2_aligned<100] = 0
                        print(mask2_aligned.shape)
                        cv2.imwrite("{}/mask/all_masks_aligned/{}/{}-{}.png".format(args.data_dir, source_name, purefilename, 4), mask2_aligned)

            #cv2.imwrite("{}/merged_img/{}/{}".format(args.data_dir, source_name, filename), im_merge)
