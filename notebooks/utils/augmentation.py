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
    
    

# def get_train_aug(RESOLUTION=300): 
#     return A.Compose([
#         A.SmallestMaxSize(max_size=RESOLUTION*2, interpolation=cv2.INTER_CUBIC, \
#                          always_apply=True),
#         A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.7, 1), \
#                             interpolation=cv2.INTER_CUBIC),
#         A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),
#         A.FancyPCA(p=0.8, alpha=0.5),
# #         A.Transpose(p=0.7),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.1),
#         A.ShiftScaleRotate(p=0.7),
#         A.HueSaturationValue(
#             hue_shift_limit=0.4, #.3
#             sat_shift_limit=0.4, #.3
#             val_shift_limit=0.4, #.3
#             p=0.7
#         ),
#         A.RandomBrightnessContrast(
#             brightness_limit=(-0.5,0.5), #-.2,.2
#             contrast_limit=(-0.4, 0.4),  #-.2,.2
#             p=0.6
#         ),
#         A.CoarseDropout(p=0.8, max_holes=30),
#         A.Cutout(p=0.8, max_h_size=40, max_w_size=40),
#         A.OneOf([
#                 A.OpticalDistortion(p=0.3),
#                 A.GridDistortion(p=.1),
#                 A.IAAPiecewiseAffine(p=0.3),
#                 ], p=0.6),
#         A.Sharpen(p=1.0, alpha=(0.1,0.3), lightness=(0.3, 0.9)),
#         A.OneOf([
#             A.IAAAdditiveGaussianNoise(p=1.0),
#             A.GaussNoise(p=1.0),
#             A.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 3))
#             ], p=0.5),
#         ], p=1.0)



def get_train_aug(RESOLUTION=300): 
    return A.Compose([
        A.LongestMaxSize(max_size=RESOLUTION*2, interpolation=cv2.INTER_CUBIC, \
                         always_apply=True),
        A.PadIfNeeded(min_height=RESOLUTION*2, min_width=RESOLUTION*2, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.7, 1), \
                            interpolation=cv2.INTER_CUBIC),
        A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),
        A.FancyPCA(p=0.8, alpha=0.5),
#         A.Transpose(p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(p=0.4, rotate_limit=12),
        A.HueSaturationValue(
            always_apply=False, p=0.3, 
            hue_shift_limit=(-20, 20), 
            sat_shift_limit=(-30, 30), 
            val_shift_limit=(-20, 20)),

#         A.HueSaturationValue(
#             hue_shift_limit=0.4, #.3
#             sat_shift_limit=0.4, #.3
#             val_shift_limit=0.4, #.3
#             p=0.7
#         ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.5,0.5), #-.2,.2
            contrast_limit=(-0.4, 0.4),  #-.2,.2
            #p=0.6
        ),
        A.CoarseDropout(p=0.8, max_holes=30),
#         A.Cutout(p=0.8, max_h_size=40, max_w_size=40),
        A.Cutout(p=1, max_h_size=60, max_w_size=30, num_holes=6, fill_value=[106,87,55]),
        A.Cutout(p=1, max_h_size=30, max_w_size=60, num_holes=6, fill_value=[106,87,55]),
        A.OneOf([
                A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.6599999666213989, 0.6800000071525574), 
                                    shift_limit=(-0.6699999570846558, 0.4599999785423279), interpolation=0, 
                                    border_mode=0, value=(0, 0, 0), mask_value=None),
#                 A.OpticalDistortion(p=0.5, distort_limit=0.15, shift_limit=0.15),
#                 A.GridDistortion(p=0.5, distort_limit=0.5),
                A.GridDistortion(always_apply=False, p=1.0, 
                                 num_steps=6, distort_limit=(-0.4599999785423279, 0.5), 
                                 interpolation=0, border_mode=0, 
                                 value=(0, 0, 0), mask_value=None),

#                 A.IAAPiecewiseAffine(p=0.5, scale=(0.1, 0.14)),
                ], p=0.6),
        A.Sharpen(p=1.0, alpha=(0.1,0.3), lightness=(0.3, 0.9)),
        A.GaussNoise(var_limit=(300.0, 500.0), p=0.4),
        A.ISONoise(always_apply=False, p=0.4, 
                   intensity=(0.10000000149011612, 1.399999976158142), 
                   color_shift=(0.009999999776482582, 0.4000000059604645)),

        A.OneOf([
            A.Equalize(always_apply=False, p=1.0, mode='cv', by_channels=True),
            A.Solarize(always_apply=False, p=1.0, threshold=(67, 120)),
#             A.IAAAdditiveGaussianNoise(p=1.0),
            A.GaussNoise(p=1.0),
            A.MotionBlur(always_apply=False, p=1.0, blur_limit=(5, 20))
            ], p=0.5),
        ], p=1.0)


# def get_train_aug(RESOLUTION=300): 
#     return A.Compose([
#         A.SmallestMaxSize(max_size=RESOLUTION*2, interpolation=cv2.INTER_CUBIC, \
#                          always_apply=True),
#         A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.7, 1), \
#                             interpolation=cv2.INTER_CUBIC),
#         A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),
#         A.FancyPCA(p=1, alpha=0.5),
#         A.CoarseDropout(p=1, max_holes=30),
#         A.Cutout(p=1, max_h_size=60, max_w_size=20, num_holes=6, fill_value=[106,87,55]),
#         A.Cutout(p=1, max_h_size=20, max_w_size=60, num_holes=6, fill_value=[106,87,55]),
#         A.Transpose(p=0.6),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.1),
#         A.ShiftScaleRotate(p=0.7),
#         A.HueSaturationValue(
#             always_apply=False, p=1, 
#             hue_shift_limit=(-40, 40), 
#             sat_shift_limit=(-50, 40), 
#             val_shift_limit=(-20, 20)),

# #         A.HueSaturationValue(
# #             hue_shift_limit=0.4, #.3
# #             sat_shift_limit=0.4, #.3
# #             val_shift_limit=0.4, #.3
# #             p=0.7
# #         ),
#         A.RandomBrightnessContrast(
#             brightness_limit=(-0.5,0.5), #-.2,.2
#             contrast_limit=(-0.5, 0.5),  #-.2,.2
#             p=1
#         ),


#         A.OneOf([
#                 A.OpticalDistortion(p=1, distort_limit=0.25, shift_limit=0.25),
# #                 A.GridDistortion(p=0.5, distort_limit=0.5),
#                 A.GridDistortion(always_apply=False, p=1, 
#                                  num_steps=6, distort_limit=(-0.2599999785423279, 0.2), 
#                                  interpolation=0, border_mode=0, 
#                                  value=(0, 0, 0), mask_value=None),

#                 A.IAAPiecewiseAffine(p=1, scale=(0.1, 0.2)),
#                 ], p=1),
#         A.Sharpen(p=1.0, alpha=(0.1,0.3), lightness=(0.3, 0.9)),
#         A.GaussNoise(var_limit=(100.2799987792969, 300.0), p=1),
#         A.ISONoise(always_apply=False, p=1, 
#                    intensity=(0.10000000149011612, 1.399999976158142), 
#                    color_shift=(0.009999999776482582, 0.4000000059604645)),

#         A.OneOf([
#             A.Equalize(always_apply=False, p=1.0, mode='cv', by_channels=True),
#             A.Solarize(always_apply=False, p=1.0, threshold=(67, 120)),
#             A.IAAAdditiveGaussianNoise(p=1.0),
# #             A.GaussNoise(p=1.0),
#             A.MotionBlur(always_apply=False, p=1.0, blur_limit=(5, 20))
#             ], p=1),
#         ], p=1.0)


def get_valid_aug(RESOLUTION=380): 
    return A.Compose([
#         A.SmallestMaxSize(max_size=RESOLUTION*2, interpolation=cv2.INTER_CUBIC, \
#                      always_apply=True),
#         A.OneOf([
#             A.CenterCrop(RESOLUTION,RESOLUTION, always_apply=True),
#             A.RandomResizedCrop(RESOLUTION,RESOLUTION, scale=(0.4, 1.0), \
#                                 always_apply=True, interpolation=cv2.INTER_CUBIC),
#             ], p=1.0),
        A.LongestMaxSize(max_size=RESOLUTION, interpolation=cv2.INTER_CUBIC, \
                         always_apply=True),
        A.PadIfNeeded(min_height=RESOLUTION, min_width=RESOLUTION, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),  
#         A.HorizontalFlip(p=0.5),
        A.FancyPCA(p=1.0, alpha=0.5),
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
