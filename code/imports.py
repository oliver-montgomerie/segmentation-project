from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    Flipd,
    LoadImaged,
    MapTransform,
    Orientationd,
    PadListDataCollate,
    Rand2DElasticd,
    RandAxisFlipd,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandRotated,
    RandZoomd,
    Rotate90d,
    ResizeWithPadOrCropd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.networks.utils import one_hot
from monai.networks.utils import predict_segmentation
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau

## Top for viewing. Below lines for saving
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import tempfile
import shutil
import os
import glob
import numpy as np
import nibabel as nib
import pickle
from skimage.measure import label as seperate_instances

min_tumor_size = 314 # pi  mm^2  for diameter 2cm tumors

def file_tumor_size(file):
    lbl = nib.load(file['label']) 
    np_lbl = np.array(lbl.dataobj)
    size_tumors = np.sum(np_lbl == 2) * lbl.header['pixdim'][1] * lbl.header['pixdim'][2]
    return size_tumors