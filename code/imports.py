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

train_files_nums = ['27', '125', '124', '113', '23', '92', '36', '120', '13', '50', '110', '126', '99', '118', '73', '59', '109', '24', '44', '29', '116', '78', '104', '31', '66', '56', '88', '43', '7', '83', '108', '40', '77', '35', '121', '86', '55', '18', '69', '70', '81', '9', '11', '22', '103', '74', '107', '58', '90', '17', '12', '26', '127', '96', '5', '101', '21', '16', '62', '39', '72', '112', '71', '6', '0', '85', '102', '3', '65', '64', '128', '122']
val_files_nums = ['75', '33', '49', '19', '61', '111', '53', '30', '28', '129', '20', '45', '51', '25', '60', '10', '84', '93', '76', '8', '97', '46', '15']
test_files_nums = ['2', '80', '117', '67', '48', '123', '94', '1', '57', '79', '95', '63', '4', '130', '68', '37', '82', '42', '14', '100', '98', '54', '52']

min_tumor_size = 314 # pi  mm^2  for diameter 2cm tumors

def file_tumor_size(file):
    lbl = nib.load(file['label']) 
    np_lbl = np.array(lbl.dataobj)
    size_tumors = np.sum(np_lbl == 2) * lbl.header['pixdim'][1] * lbl.header['pixdim'][2]
    return size_tumors



