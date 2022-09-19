# read from original files
#
# dataset:
#   -train
#       -images
#       -labels
#   -val
#       -images
#       -labels

from torch import rand
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
import cv2

from data.utils import *

class Data(Dataset):
    def __init__(self, root, args, folder):
        self.folder = os.path.join(root, folder)
        self.collect_data_names()
        print('number of '+folder+' images:', len(self.name_list))

        self.channel = args.channel
        self.aug = args.aug
        self.norm  =args.norm
        if isinstance(args.imgsz, int):
            self.size = [args.imgsz, args.imgsz]
        else:
            self.size = [x for x in args.imgsz]
    
    # collect all data names and store in a list
    def collect_data_names(self):
        self.name_list = []
        img_path = os.path.join(self.folder, 'images')
        for img in tqdm(os.listdir(img_path)):
            name = img.split('.')[0]
            label_path = os.path.join(self.folder, 'labels/'+name+'.png')
            if not os.path.exists(label_path):
                continue
            self.name_list.append(name)
        random.shuffle(self.name_list)

    # load image and label based on the data name, and data augmentation if necessary
    def load_data(self, name):
        img_path = os.path.join(self.folder, 'images/'+name+'.jpg')
        label_path = os.path.join(self.folder, 'labels/'+name+'.png')

        # read image and label
        x = Image.open(img_path).convert('RGB')
        x = np.asarray(x)

        y = Image.open(label_path)
        y = np.asarray(y)
        if len(y.shape) == 3:
            y = y[..., 0]
        y = y/y.max() # label for each pixel would be 0-1, not binary, some values would be not 0 or 1

        # data augmentation
        if self.aug:
            x, y = random_crop(x, y)
            x, y = random_rotate(x, y)
            x = random_light(x)

        # transfer RGB to grayscale image
        if self.channel != 'rgb':
            x = c2g(x)
        
        # normalize data [rgb/grayscale]
        if self.norm and self.channel == 'rgb':
            x[..., 0] -= 123.68
            x[..., 1] -= 116.779
            x[..., 2] -= 103.939

        # resize data and label
        x = cv2.resize(x, self.size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        y = cv2.resize(y, self.size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        y = y.reshape((1, self.size[0],self.size[1]))
        if self.channel != 'rgb':
            x = x.reshape((1, self.size[0],self.size[1]))
        else:
            x = np.transpose(x, (2, 0, 1))
        return x, y
        

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        return self.load_data(name)