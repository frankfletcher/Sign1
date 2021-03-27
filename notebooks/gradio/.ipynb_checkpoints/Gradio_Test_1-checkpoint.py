#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')  # or 'once'  

import pyforest
import gradio as gr

from datetime import datetime
from fastbook import *
from fastai.vision.all import *
from fastai.vision.widgets import *
import fastai
import gc
# from nn_utils_eff import *
# from augmentation import *

from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from timm import create_model
from fastai.vision.learner import _update_first_layer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from string import ascii_uppercase
import numpy as np
import operator
import glob

import torch
from torch import nn as nn
from torch.nn import functional as F

import albumentations as A
import cv2

########################################### contents of nn_utils_eff and augmentation






def show_cuda_status():
    import torch
    print('CUDA available: '.ljust(28), torch.cuda.is_available())
    print('CUDA device count: '.ljust(28), torch.cuda.device_count())

    current_device = torch.cuda.current_device()
    print('Current CUDA Device index: '.ljust(28), current_device)
    # torch.cuda.device(current_device)
    print('Current CUDA Device: '.ljust(28), torch.cuda.get_device_name(current_device))
    print()
    # print('CUDA available: '.ljust(24), torch.cuda.is_available())
    print(f'fastai version:              {fastai.__version__}')
    # print(f'fastcore version:            {fastcore.__version__}')
    # print(f'fastbook version:            {fastbook.__version__}')
    print(f'cuda version:                {torch.version.cuda}')
    print(f'torch version:               {torch.__version__}')
    # print(f'python version:              {python_version()}')
    
    


# timm library from Ross Wightman
# integration of timm w/ fastai from Zach Mueller

def create_timm_body(arch:str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or function")
        
def create_timm_model(arch:str, n_out, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_, custom_head=None,
                     concat_pool=False, **kwargs):
    "Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library"
    body = create_timm_body(arch, pretrained, None, n_in)
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else: head = custom_head
    model = nn.Sequential(body, head)
    if init is not None: apply_init(model[1], init)
    return model


def timm_learner(dls, arch:str, loss_func=None, pretrained=True, cut=None, splitter=None,
                y_range=None, config=None, n_out=None, normalize=True, custom_head=None, **kwargs):
    "Build a convnet style learner from `dls` and `arch` using the `timm` library"
    if config is None: config = {}
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = create_timm_model(arch, n_out, default_split, pretrained, y_range=y_range, custom_head=custom_head, **config)
    learn = Learner(dls, model, loss_func=loss_func, splitter=default_split, **kwargs)
    if pretrained: learn.freeze()
    return learn

import kornia
def convert_MP_to_blurMP(model, layer_type_old):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_MP_to_blurMP(module, layer_type_old)

        if type(module) == layer_type_old:
            layer_old = module
            layer_new = kornia.contrib.MaxBlurPool2d(3, True)
            model._modules[name] = layer_new

    return model


def convert_act_cls(model, layer_type_old, layer_type_new):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_act_cls(module, layer_type_old, layer_type_new)

        if type(module) == layer_type_old:
            layer_old = module
            layer_new = layer_type_new
            model._modules[name] = layer_new

    return model






# as far as I can track back, the origination of this CheckpointModule is from Elad Hoffer: 
# https://github.com/eladhoffer/convNet.pytorch/blob/master/models/modules/checkpoint.py

class CheckpointModule(Module):
    def __init__(self, module, num_segments=1):
        assert num_segments == 1 or isinstance(module, nn.Sequential)
        self.module = module
        self.num_segments = num_segments

    def forward(self, *inputs):
        if self.num_segments > 1:
            return checkpoint_sequential(self.module, self.num_segments, *inputs)
        else:
            return checkpoint(self.module, *inputs)
        
        
    
def convert_seq_chkpt(model, layer_type_old=nn.Sequential):  
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_seq_chkpt(module, layer_type_old) 

        if type(module) == layer_type_old:
#             num_chkpt += 1
            layer_old = module
#             if len(layer_old) == 7:
#                 segments = 1
#             else: segments = len(layer_old)
            segments = len(layer_old)
            layer_new = CheckpointModule(layer_old, segments)  # wrap sequential in a checkpoint module
            model._modules[name] = layer_new
#             print(segments)

    return model






class SwishAutoFn(torch.autograd.Function):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    Memory efficient variant from:
     https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
    """
    @staticmethod
    def forward(ctx, x):
        result = x.mul(torch.sigmoid(x))
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_sigmoid = torch.sigmoid(x)
        return grad_output.mul(x_sigmoid * (1 + x * (1 - x_sigmoid)))


def swish_auto(x, inplace=False):
    # inplace ignored
    return SwishAutoFn.apply(x)


class SwishAuto(nn.Module):
    def __init__(self, inplace: bool = True):
        super(SwishAuto, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return SwishAutoFn.apply(x)




class MishAutoFn(torch.autograd.Function):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    Experimental memory-efficient variant
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_sigmoid = torch.sigmoid(x)
        x_tanh_sp = F.softplus(x).tanh()
        return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


def mish_auto(x, inplace=False):
    # inplace ignored
    return MishAutoFn.apply(x)


class MishAuto(nn.Module):
    def __init__(self, inplace: bool = True):
        super(MishAuto, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return MishAutoFn.apply(x)
    
    
    
    
    
def remove_cbs(le):
    _cbs = le.cbs
    
    le.remove_cbs(CutMix)
    le.remove_cbs(SaveModelCallback)
    le.remove_cbs(ShowGraphCallback)
    le.remove_cbs(EarlyStoppingCallback)
    le.remove_cbs(GradientAccumulation)
    return _cbs





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
#         A.Resize(RESOLUTION, RESOLUTION, p=1.0, interpolation=cv2.INTER_CUBIC),  
        A.HorizontalFlip(p=0.5),
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



##############################################




# RUN_NAME = '20210323-1337 - arch=tf_efficientnet_b4_ns - samples=7500 frozen=1 epochs=10 bs=48 res=380 _data=combined4_with_overflow_all_d'

RUN_NAME = '20210323-2340 - arch=tf_efficientnet_b4_ns - samples=7500 frozen=1 epochs=10 bs=48 res=380 _data=combined4_with_overflow_all_d'


# with help from Ali Abid @ Gradio
dir = os.path.dirname(__file__)
RUN_NAME = os.path.join(dir, RUN_NAME)
learn_inf = load_learner(f'{RUN_NAME}.pkl', cpu=True)


#'20210303-1849 - arch=tf_efficientnet_b4_ns - samples=1100 frozen=1 epochs=15 bs=48 res=380 _data=combined2'

learn_inf = load_learner(f'{RUN_NAME}.pkl', cpu=True)
remove_cbs(learn_inf)
# print('Model Loaded.')

learn_inf.remove_cbs([NonNativeMixedPrecision, ModelToHalf])

labels = [char for char in ascii_uppercase]
labels.remove('J')
labels.remove('Z')

def get_sign(img):
    pred,pred_idx,probs = learn_inf.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
 
image = gr.inputs.Image(shape=(380,380))
label = gr.outputs.Label(num_top_classes=3)
iface = gr.Interface(fn=get_sign, inputs=image, outputs=label)
iface.launch(share=True)

