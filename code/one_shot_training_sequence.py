import numpy as np
import cv2
# from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import Sequence
from imgaug import augmenters as iaa
import random
import utils
import glob

class OneShotTrainingSequence(Sequence):
    def __init__(self, image_dir, mask_dir, da=False, batch_size=1, image_width=64, image_height=64):
        self.image_width = image_width
        self.image_height = image_height
        self.da = da
        self.image_list, self.mask_list, self.file_name_list = \
                self.load_image_and_mask(image_dir, mask_dir)
        self.batch_size = batch_size
        self.channel_number = 3
        self.seq = iaa.Sequential([
            iaa.Add((-10, 10))])

    def load_image_and_mask(self, image_dir, mask_dir):
        image_list = []
        mask_list = []
        file_name_list = []
        for img_name in glob.glob(image_dir + '/*.png'):
            pure_name = img_name.split('/')[-1]
            pure_name = pure_name.split('.')[0]
            file_name_list.append(pure_name)
            im = cv2.imread(img_name, cv2.IMREAD_COLOR)
            im = cv2.resize(im, (self.image_width, self.image_height))
            # aug_image = cv2.equalizeHist(im_gray)
            aug_image = im
            image_list.append(aug_image)
            mask = cv2.imread(mask_dir + '/{}.png'.format(pure_name), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_height, self.image_width), interpolation=cv2.INTER_NEAREST)
            mask_list.append(mask)

        return np.asarray(image_list), np.asarray(mask_list), file_name_list

    def __len__(self):
        return int(self.image_list.shape[0] / float(self.batch_size))

    def __getitem__(self, idx):
        idx = random.randint(0,self.image_list.shape[0]-1)
        batch_x = np.zeros(shape=(self.batch_size, self.image_width, self.image_height, 3), dtype=np.float32)
        batch_y = np.zeros(shape=(self.batch_size, self.image_width, self.image_height, 1), dtype=np.uint8)

        for i in range(self.batch_size):
            if self.da:
                #augmentation
                if random.random() < -0.7:
                    angle = 0
                else:
                    #angle = random.uniform(-25, 25)
                    angle = random.uniform(-10, 10)
                if random.random() < -0.5:
                    scale = 1.0
                else:
                    #scale = random.uniform(0.9, 1.1)
                    scale = random.uniform(0.95, 1.05)
                tmp_rotation_matrix = cv2.getRotationMatrix2D((self.image_width/2, self.image_height/2),
                                                              angle=angle, scale=scale)
                rotation_matrix = np.eye(3, dtype=np.float32)
                rotation_matrix[0:2, :] = tmp_rotation_matrix
                
                shearing_matrix = np.eye(3, dtype=np.float32)
                if random.random() > -0.5:
                    shearing_matrix[0,1] = 0.0
                    shearing_matrix[1,0] = 0.0
                else:
                    shearing_matrix[0,1] = random.uniform(-0.005, 0.005)
                    shearing_matrix[1,0] = random.uniform(-0.005, 0.005)

                translation_matrix = np.eye(3, dtype=np.float32)
                translation_matrix[0,2] = random.randint(-10, 10)
                translation_matrix[1,2] = random.randint(-10, 10)
                
                transform_matrix = np.matmul(translation_matrix, np.matmul(shearing_matrix, rotation_matrix))

                transformed_image = cv2.warpPerspective(self.image_list[idx], transform_matrix, (self.image_width, self.image_height),\
                        flags=cv2.INTER_LINEAR, borderValue = (255,255,255))
                transformed_mask = np.zeros((self.image_height, self.image_width, 1), dtype = np.uint8)

                temp_mask = cv2.warpPerspective(self.mask_list[idx], transform_matrix, (self.image_width, self.image_height),\
                        flags=cv2.INTER_NEAREST, borderValue = (0))
                transformed_mask[temp_mask>100] = 255

                aug_image = self.seq.augment_image(transformed_image)
                #show_mask = utils.drawMultiRegionMultiChannel(transformed_mask)
                ##aug_image = cv2.equalizeHist(aug_image)
                #cv2.imwrite('../data/augmentation/{}_img.png'.format(i), aug_image)
                #cv2.imwrite('../data/augmentation/{}_mask.png'.format(i), show_mask)
                
                batch_x[i] = aug_image
                batch_y[i] = transformed_mask
            else:
                batch_x[i] = self.image_list[idx]
                batch_y[i,:,:,0] = self.mask_list[idx]
        
        batch_x = batch_x/255.0
        batch_y[batch_y<100] = 0
        batch_y[batch_y>=100] = 1

        return batch_x, batch_y

