Sign1
==============================

Classification of the ASL Alphabet with deployment to webcam ("realtime" video) as well as an interactive online demo (images) at https://gradio.app/g/cogsci2/Sign1

![Video of ASL classification](https://github.com/cogsci2/Sign1/blob/master/reports/videos/20210325%20Sign1%20Webcam%20test1-2021-03-25_15.29.20.gif)

<br>

## The Challenge: To create a model to translate ASL alphabet signs to written characters

The technical goal was to check the viability of using sota techniques to translate ASL into written English characters.  Based on the success of this experiment, we think it may be possible to use naive-but-modern CNN models for ASL recognition.

<br>

## Project Status:  Backburner.  
### (Prototype Successful)

Our latest model uses an Efficientnet b4, with tensorflow pretrained weights.  We are able to achieve 92% on a third party dataset that was designed specifically to challenge ASL alphabet classifiers. 

A slide overview of the project is available here: https://docs.google.com/presentation/d/1CGssA6PaNyEU4xf-YNqp3lbroWTFlCutIiDqM3pc6yE/edit?usp=sharing


<br>

## Hardware Used:  RTX 2070 Super (8gb RAM)

Using a GPU with only 8gb of RAM presented it's own challenges when working with large modern CNN architectures.  To overcome this limitation we used several techniques:

1. We developed a fairly novel technique to recursively edit a pre-existing model - wrapping `nn.Sequential` with checkpoint modules.  This allowed us to tack on Gradient Checkpoints to almost any pre-loaded model organized with nn.Sequential.  We were able to successfully deploy this technique on multiple ResNet variations as well as EfficientNet and DenseNet.
2. We use mixed precision training.
3. We use gradient accumulation
4. In order to tune the learning rate, we temporarily removed gradient accumulation, and decreased the resolution while increasing the batch size.  Once we were happy with the LR, we revert the resolution and batch sizes back so it can fit on the GPU and restart the training process.

<br>

## Best Model (so far):  pretrained EfficientNet b4
(https://github.com/cogsci2/Sign1/blob/master/notebooks/Archive/20210323-2340%20-%20Sign4%20-%20EfficientNet-93perc.ipynb)

Some Notes on this franken-model:
1. see the "Hardware Used" section for how we optimized the model for an older GPU with only 8GB RAM.
2. We recursively replaced the stock SiLU (Swish) activation functions with a memory friendly version (SwishAuto)
3. Due to the overhead, we use Mish activation only on our custom heads hidden layer.
4. We built a custom head in order to tune the output and prevent over-fitting.  We narrow the channels substantially as well as increase dropout.  
5. We use a substantial amount of augmentation and other techniques in order to obfuscate the handsigns during training and force the model to "stretch".  One technique we used was to create shaped (image) dropouts, roughly the size of fingers and colored in skin-tones. 
6. Partly because of our challenges with homogenous data, we use many regularization techniques:  CutMix, Label Smoothing, dropout, etc

We achieve similar results (~92%) using EfficientNet Lite4 which is a more feasible model to deploy to mobile phones. (https://github.com/cogsci2/Sign1/blob/master/notebooks/Archive/Sign4%20-%20EfficientNetLITE.ipynb)

<br>

## Modern Techniques Used

We used many state of the art techniques as well as attempting over 50 training runs using 9 different architectures.  In the end, we decided on a modified EfficientNet architecture.  Some of the modifications are listed here: 

* Gradient Checkpointing --> Low-Memory Neural Network Training: A Technical Report 2019 (https://arxiv.org/abs/1904.10631)
* MaxBlurPool Layer --> Making Convolutional Networks Shift-Invariant Again 2019 (https://arxiv.org/abs/1904.11486)
* Mish Activation Function --> Mish: A Self Regularized Non-Monotonic Activation Function 2019 (https://arxiv.org/abs/1908.08681)
* Swish Activation Function --> Searching for Activation Functions 2017 (https://arxiv.org/abs/1710.05941v2)
* Rectified Adam Optimizer --> On the Variance of Adaptive Learning Rate and Beyond 2019 (https://arxiv.org/abs/1908.03265)
* Look Ahead Optimizer --> Lookahead Optimizer: k steps forward, 1 step back 2019 (https://arxiv.org/abs/1907.08610)
* Label Smoothing --> Regularizing Neural Networks by Penalizing Confident Output Distributions 2017 (https://arxiv.org/abs/1701.06548)
* CutMix --> CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features 2019 (https://arxiv.org/abs/1905.04899)

<br>

## Data - External

About half of our data was obtained externally from the following sources:
https://www.kaggle.com/grassknoted/asl-alphabet
https://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset

Our primary challenge holdout set was created by Dan Rasband.  This set was quite interesting as many of the signs were made at what looks to be a construction site with varied backgrounds as well as occasionally strong backlighting.  This could easily fool a model into thinking the background was relevent to the sign.
https://www.kaggle.com/danrasband/asl-alphabet-test

## Data - Internal

We also set out to create our own data.  We developed a technique to use a webcam to capture image frames and label them - this was done by pressing a character on the keyboard while making a sign.  The image was saved to disk and automatically labelled based on the character that was pressed.  

Using this technique, we were able to create images at a rate of about 8 frames per second but there were issues.  The webcam interface we used didn't allow us to move from one position to another quickly enough; anytime we moved too fast, we had many images with motion blur. 

Without the ability to move from position to position, the images obtained would be even more homogenous.  Because of that, we found it more effective to use a cell phone to video the sign while continuously moving, changing backgrounds, lighting, and camera angles, and sign variations.  We could even walk through different locales while holding the sign.

To process these videos, we developed several small utilities:
1. Loop through all character videos and explode them into images (https://github.com/cogsci2/Sign1/tree/master/notebooks/utils/ExplodeVideo.ipynb)
2. Create a new sub-datgaset from a larger dataset: (https://github.com/cogsci2/Sign1/tree/master/notebooks/utils/DatasetCuller.ipynb)

as well as some others.

We tried hard to vary our internal data as much as we could within our means:
* 3 different people with different skin tones and both sexes represented
* Multi colored indoor lighting - tungsten, flourescent, etc
* Very bright sunlight, outdoors
* Very strong backlighting
* With almost no ambient light
* illumination from only a laptop screen
* mutiple colored kitchen gloves (decrease the dependency on skin tones)
* signing in front of a desktop screen with many hands in the background

## Data - Augmentation / Obfuscation

We really tried to challenge the models.  Most of the time, the models barely blinked at our obfuscation efforts.  

<br>

## Model Deployment: Webcam and Gradio

We developed a webcam deployment, using OpenCV.  This deployment allows for semi-realtime interaction with the model.  (https://github.com/cogsci2/Sign1/blob/master/notebooks/OpenCV_cam_test.ipynb)

We also deployed the model to the web, using Gradio.  This allows anybody with a web browswer (desktop, laptop or cel phone) the ability to upload a snapshot and get the English translation back.  Basically, this is a dictionary-type reference system.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebook source code. 
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


