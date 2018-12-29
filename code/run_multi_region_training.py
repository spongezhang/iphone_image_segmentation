"""
Run multiple parameter with multiple GPUs and one python script 
Usage: python run_all.py

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

#! /usr/bin/env python2

import numpy as np
import scipy.io as sio
import time
import os
import sys
import subprocess
import shlex
import argparse


####################################################################
# Parse command line
####################################################################
def usage():
    print >> sys.stderr 
    sys.exit(1)


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

#define gpu you want to use
gpu_set = ['0']
#gpu_set = ['0', '1', '2', '3'] #if you want to use more

parameter_set = [\
        # one single region test
        #'--region_list=4 --loss=dice --metric=iou-multi-region ',
        '--loss=dice --metric=iou ',
        #'--region_list=4 --loss=dice --metric=iou-multi-region --no_alignment --no_ref_mask ',
        ]

number_gpu = len(gpu_set)
process_set = []

for run in range(1):
    for idx, parameter in enumerate(parameter_set):
        print('Test Parameter: {}'.format(parameter))
        command = 'python one_shot_training_multi_region.py  --data_dir=../data/  --log_dir=../result/unet_log/ \
                --output_dir=../result/result/ --model_dir=../result/model/ \
                 {} --num_epochs=100 \
                --gpu-id {} --idx={} '.format(parameter, gpu_set[idx%number_gpu], run)

        print(command)
        p = subprocess.Popen(shlex.split(command))
        process_set.append(p)
        
        if (idx+1)%number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()
        
            process_set = []
        time.sleep(10)

for sub_process in process_set:
    sub_process.wait()

