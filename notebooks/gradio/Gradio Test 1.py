#!/usr/bin/env python
# coding: utf-8

# import sys
# sys.path.append('../utils')


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
from nn_utils_eff import *
from augmentation import *




RUN_NAME = '20210303-1849 - arch=tf_efficientnet_b4_ns - samples=1100 frozen=1 epochs=15 bs=48 res=380 _data=combined2'

learn_inf = load_learner(f'{RUN_NAME}.pkl', cpu=True)
remove_cbs(learn_inf)
print('Model Loaded.')


learn_inf.remove_cbs([NonNativeMixedPrecision, ModelToHalf])
learn_inf.cbs


labels = [char for char in ascii_uppercase]
labels.remove('J')
labels.remove('Z')


def get_sign(img):
    pred,pred_idx,probs = learn_inf.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
 
image = gr.inputs.Image(shape=(380,380))
label = gr.outputs.Label(num_top_classes=3)
iface = gr.Interface(fn=get_sign, inputs=image, outputs=label)
iface.launch()




