import os
import random
import cv2
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from utils import *
from .augment import *

### copy from https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
class CutoutV2(A.DualTransform):
    def __init__(
        self,
        num_holes=8,
        max_h_size=8,
        max_w_size=8,
        fill_value=0,
        always_apply=False,
        p=0.5,
    ):
        super(CutoutV2, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def apply(self, image, fill_value=0, holes=(), **params):
        return A.functional.cutout(image, holes, fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(self.num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y1 + self.max_h_size, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x1 + self.max_w_size, 0, width)
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size")   
    
def get_train_transforms(CFG):
    return A.Compose([   
            ##copy from https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
            A.ImageCompression(quality_lower=80, quality_upper=99),
            A.Resize(CFG.image_size, CFG.image_size),
            # A.RandomResizedCrop(CFG.image_size, CFG.image_size, scale=(0.6, 1), ratio=(0.5, 2)),
            A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.OneOf([
#                     A.OpticalDistortion(distort_limit=1.),
#                     A.GridDistortion(num_steps=5, distort_limit=1.),
#                 ], p=0.25),
            A.RandomBrightness(limit=0.2, p=0.75),
            A.RandomContrast(limit=0.2, p=0.75),
            A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=10, border_mode=0, p=0.75),
#             A.RandomSunFlare(p=0.1),
#             A.RandomFog(p=0.5),
#             A.Rotate(p=0.5, limit=90),
#             A.RGBShift(p=0.5),
#             A.RandomSnow(p=0.1),
#             A.Blur(p=0.2),
            CutoutV2(max_h_size=int(CFG.image_size * 0.4), max_w_size=int(CFG.image_size * 0.4), num_holes=1, p=0.5),
            ###for cls without mask
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ],p=1.0)

def get_val_transforms(CFG):
    return A.Compose([
            A.Resize(CFG.image_size, CFG.image_size),
            ###for cls without mask
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],p=1.0)
    
class TrainMultiHeadDataset(Dataset):
    def __init__(self, CFG, df, transforms=None):
        super().__init__()
        self.CFG = CFG
        self.df = df
        self.transforms = transforms
        self.length = len(df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        if self.CFG.do_crop:
            image_path = self.CFG.crop_train_dir + '%s' % (d.image) ### crop images with yolo
        elif self.CFG.do_mask:
            image_path = self.CFG.crop_train_dir + '%s' % (d.image).replace('.jpg','.png') ### crop images with tracer
        else:
            image_path = self.CFG.train_dir + '%s' % (d.image)   ### images from source
        
        ### load image
        image = cv2.imread(image_path)        
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ### load label
        # label = self.CFG.labels_mapping[d.label]
        # species = self.CFG.species_mapping[d.species]
        individual_id = self.CFG.individual_ids_mapping[d.individual_id]
        
        # apply augmentations
        if self.transforms:
            image = self.transforms(image=image)['image']
        else:
            image = torch.from_numpy(image)
        # return image, label, species, individual_id
        return image, individual_id
    
    
class ValidMultiHeadDataset(Dataset):
    def __init__(self, CFG, df, transforms=None):
        super().__init__()
        self.CFG = CFG
        self.df = df
        self.transforms = transforms
        self.length = len(df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        if self.CFG.do_crop:
            image_path = self.CFG.crop_train_dir + '%s' % (d.image) ### crop images with yolo
        elif self.CFG.do_mask:
            image_path = self.CFG.crop_train_dir + '%s' % (d.image).replace('.jpg','.png') ### crop images with tracer
        else:
            image_path = self.CFG.train_dir + '%s' % (d.image)   ### images from source
        
        ### load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ### load label
        image_id = d.image
        # label = self.CFG.labels_mapping[d.label]
        # species = self.CFG.species_mapping[d.species]
        individual_id = self.CFG.individual_ids_mapping[d.individual_id]
        
        # apply augmentations
        if self.transforms:
            image = self.transforms(image=image)['image']
        else:
            image = torch.from_numpy(image)
        return image, individual_id, image_id
    
class TestDataset(Dataset):
    def __init__(self, CFG, df, transforms=None):
        super().__init__()
        self.CFG = CFG
        self.df = df
        self.transforms = transforms
        self.length = len(df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        image_id = d.image
        
        image_path = self.CFG.crop_test_dir + '%s' % (d.image) ### crop images with detic
        ### load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # apply augmentations
        if self.transforms:
            image = self.transforms(image=image)['image']
        else:
            image = torch.from_numpy(image)
        return image, image_id
