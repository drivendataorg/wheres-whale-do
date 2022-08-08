import os
import random
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

import torch
import torch.nn as nn

import logging
def get_log(file_name):
    logger = logging.getLogger('train')  
    logger.setLevel(logging.INFO)  
 
    ch = logging.StreamHandler()  
    ch.setLevel(logging.INFO) 
 
    fh = logging.FileHandler(file_name, mode='a')  
    fh.setLevel(logging.INFO)  

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)  
    fh.setFormatter(formatter)
    logger.addHandler(fh)  
    logger.addHandler(ch)
    return logger

def tensor2im(input_image, imtype=np.uint8):
    mean = [0.485,0.456,0.406] #Ã—Ã”Â¼ÂºÃ‰Ã¨Ã–ÃƒÂµÃ„
    std = [0.229,0.224,0.225]  #Ã—Ã”Â¼ÂºÃ‰Ã¨Ã–ÃƒÂµÃ„
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_img(im, path, size):
    if os.path.isfile(path):  # do not overwrite
        return None
    im_grid = torchvision.utils.make_grid(im, size) 
    im_numpy = tensor2im(im_grid) 
    im_array = Image.fromarray(im_numpy)
    im_array.save(path)