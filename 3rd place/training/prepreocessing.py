import glob
import cv2
from skimage.io import imread, imsave
import os


for im in os.listdir('input/train_images'):
    img = imread('./input/train_images/' + im)
    imsave('./input/train_images_new/' + im, cv2.resize(img, (640, 640)))