import albumentations as A
from fastai.vision import *
from fastbook import *
from fastai.vision.all import *
from fastai.vision.widgets import *
import cv2


class AlbumentationsTransform(RandTransform):
    "A transform handler for multiple `Albumentation` transforms"
    split_idx,order=None,2
    def __init__(self, train_aug, valid_aug): store_attr()
    
    def before_call(self, b, split_idx):
        self.idx = split_idx
    
    def encodes(self, img: PILImage):
        if self.idx == 0:
            aug_img = self.train_aug(image=np.array(img))['image']
        else:
            aug_img = self.valid_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)
    
    

def get_train_aug(RESOLUTION=300): 
    return A.Compose([
        A.SmallestMaxSize(max_size=RESOLUTION*2, interpolation=cv2.INTER_CUBIC, \
                         always_apply=True),
        A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.7, 1), \
                            interpolation=cv2.INTER_CUBIC),
        A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),
        A.FancyPCA(p=0.8, alpha=0.5),
#         A.Transpose(p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(p=0.7),
        A.HueSaturationValue(
            hue_shift_limit=0.3, 
            sat_shift_limit=0.3, 
            val_shift_limit=0.3, 
            p=0.7
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2,0.2), 
            contrast_limit=(-0.2, 0.2), 
            p=0.7
        ),
        A.CoarseDropout(p=0.8, max_holes=30),
        A.Cutout(p=0.8, max_h_size=40, max_w_size=40),
        A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
                ], p=0.6),
        A.Sharpen(p=1.0, alpha=(0.1,0.3), lightness=(0.3, 0.9)),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(p=1.0),
            A.GaussNoise(p=1.0),
            A.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 3))
            ], p=0.5),
        ], p=1.0)



def get_valid_aug(RESOLUTION=300): 
    return A.Compose([
#         A.SmallestMaxSize(max_size=RESOLUTION*2, interpolation=cv2.INTER_CUBIC, \
#                      always_apply=True),
#         A.OneOf([
#             A.CenterCrop(RESOLUTION,RESOLUTION, always_apply=True),
#             A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.4, 1.0), \
#                                 always_apply=True, interpolation=cv2.INTER_CUBIC),
#             ], p=1.0),
        A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),  
        A.HorizontalFlip(p=0.5),
        A.FancyPCA(p=0.75, alpha=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.1, 
            sat_shift_limit=0.1, 
            val_shift_limit=0.1, 
            p=0.5
            ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1,0.1), 
            contrast_limit=(-0.1, 0.1), 
            p=0.5
            ),
        A.Sharpen(p=1.0, alpha=(0.1, 0.3), lightness=(0.3, 0.9))
        ], p=1.0)
