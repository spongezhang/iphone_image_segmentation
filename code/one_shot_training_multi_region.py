from __future__ import print_function
import shutil
import os
import keras.backend as k
import utils
from one_shot_training_sequence import OneShotTrainingSequenceMultiRegion
import train_utils
import U_net
import cv2
import numpy as np
import keras.callbacks
import csv
import argparse
from keras.utils.data_utils import OrderedEnqueuer


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', default='../data/', help='folder to image data')
parser.add_argument('--log_dir', default='../result/unet_log/', help='folder to output log')
parser.add_argument('--model_dir', default='../result/model/', help='folder to output log')
parser.add_argument('--output_dir', default='../result/image/', help='folder to output log')

parser.add_argument('--unet_depth', default=4, type=int, help='show step')
parser.add_argument('--start_filters', default=8, type=int, help='show step')

parser.add_argument('--loss', default='iou', type=str)
parser.add_argument('--metric', default='iou', type=str)

parser.add_argument('--idx', default=0, type=int, help='for multi test')

parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=4, type=int, help='batch size for training')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epoch for training')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# hyper-parameters
image_size = (75, 75)
data_format = 'channels_last'
monitor = 'val_loss'

suffix = args.source_name + '_{}'.format(args.loss)

channels = 3

k.set_image_data_format(data_format)

training_image_dir = '{}/crop_train/'.format(args.data_dir)
training_mask_dir = '{}/train_label/'.format(args.data_dir)
tmp_training_sequence = OneShotTrainingSequence(training_image_dir, training_mask_dir, batch_size=4)
training_sequence = OrderedEnqueuer(tmp_training_sequence, shuffle = True)

test_image_dir = '{}/crop_validation/'.format(args.data_dir)
test_mask_dir = '{}/validation_label/'.format(args.data_dir)
validation_sequence = OneShotTrainingSequence(test_image_dir, test_mask_dir, batch_size=1)

# build model
model = U_net.UNET()
if 'cross' in args.loss:
    use_softmax = True
else:
    use_softmax = False

input_channels = channels

u_net = model.BuildUnet((image_size[1], image_size[0], input_channels), args.start_filters, args.unet_depth,\
        dropout=False, _nchannel = channels, use_softmax = use_softmax)

train_utils.Compile(u_net, optimizer='Adam', lr=args.learning_rate, metric=args.metric, loss=args.loss)

try:
    os.stat('{}/{}/'.format(args.model_dir, suffix))
except:
    os.makedirs('{}/{}/'.format(args.model_dir, suffix))

model_check_point = keras.callbacks.ModelCheckpoint("{}/{}/best_model.h5".format(args.model_dir, suffix),\
        monitor=monitor, save_best_only=True, save_weights_only=True)

try:
    os.stat('{}/{}'.format(args.log_dir, suffix))
    shutil.rmtree('{}/{}'.format(args.log_dir, suffix)) 
except:
    pass

tensorboard = keras.callbacks.TensorBoard(log_dir='{}/{}'.format(args.log_dir, suffix), batch_size=args.batch_size)

u_net.fit_generator(
        train_sequence,
        validation_data=validation_sequence,
        validation_steps=len(validation_sequence),
        steps_per_epoch=len(train_sequence.sequence)/args.batch_size,
        epochs=args.num_epochs,
        verbose=1,
        callbacks=[model_check_point, tensorboard],
    )

# for final test
#try:
#    os.stat('{}/{}/'.format(args.output_dir, suffix))
#except:
#    os.makedirs('{}/{}/'.format(args.output_dir, suffix))
#
#u_net.load_weights("{}/{}/best_model.h5".format(args.model_dir, suffix))
#
#filename_list = []
#all_iou_list = []
#
#for i in range(len(test_sequence)):
#    #get predict
#    output = u_net.predict(test_sequence[i][0], batch_size=args.batch_size)
#    output = output[0,:,:,:]
#    
#    index_output = np.argmax(output, axis = 2)
#
#    show_mask = utils.drawMultiRegionIndex(index_output)
#    cv2.imwrite('{}/{}/{}.png'.format(args.output_dir, suffix,\
#            test_sequence.file_name_list[i]), show_mask)
