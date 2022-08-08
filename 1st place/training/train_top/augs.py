import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from config import CFG
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy
import cv2
import matplotlib.pyplot as plt

def triangle(img, p):
    xx = numpy.random.rand(1)[0]
    if xx > p:
        h, w, _= img.shape
        limitw = int(w * 0.3)
        limith = int(h * 0.25)
        desc = 0
        step = limitw / limith
        for i in range(limith):
            img[i][:limitw - int(step * i)] = (255, 255, 255)
    return img

class Triangle(ImageOnlyTransform):
    def __init__(self, p):
        super(Triangle, self).__init__(p)
        self.p = p
    def apply(self, img , **params):
        return triangle(img , self.p)

def get_train_transforms():
    return albumentations.Compose(
        [   
            albumentations.Resize(CFG.DIM[0],CFG.DIM[1],always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.OneOf([
            albumentations.Sharpen(p=0.3),
            albumentations.ToGray(p=0.3),
            albumentations.CLAHE(p=0.3),
            ], p=0.5),
            albumentations.ShiftScaleRotate(
               shift_limit=0.25, scale_limit=0.2, rotate_limit=4,p = 0.5
            ),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            albumentations.Normalize(),
            ToTensorV2(),
        ]
    )

def get_train_transforms2():
    print("Test Triangle Augs")
    img = cv2.imread("../data/images/train0000.jpg")
    f = Triangle(p = 0.5)
    out = f(image=img)
    plt.imshow(out["image"])
    return albumentations.Compose(
        [   Triangle(p = 0.5),
            albumentations.Resize(CFG.DIM[0],CFG.DIM[1],always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.OneOf([
            albumentations.Sharpen(p=0.3),
            albumentations.ToGray(p=0.3),
            albumentations.CLAHE(p=0.3),
            ], p=0.5),
            albumentations.ShiftScaleRotate(
               shift_limit=0.25, scale_limit=0.2, rotate_limit=4,p = 0.5
            ),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            albumentations.Normalize(),
            ToTensorV2(),
        ]
    )
def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(CFG.DIM[0],CFG.DIM[1],always_apply=True),
            albumentations.Normalize(),
        ToTensorV2(p=1.0)
        ]
    )