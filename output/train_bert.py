import os
import gc
import sys
import cv2
import math
import time
import tqdm
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader

from transformers import (AutoModel, AutoTokenizer,
                          AutoModelForSequenceClassification)
from transformers import (RobertaTokenizer, RobertaModel)

from utils import dotdict


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    config = dotdict({
        'seed': 42,
        'batch_size': 10,
        'lr': 1e-5,
        'max_len': 64,
        'train_data_path': '../input/commonlitreadabilityprize/train.csv',
        'test_data_path': '../input/commonlitreadabilityprize/test.csv',
        'sample_path': '../input/commonlitreadabilityprize/sample_submission.csv'
    })

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=config.seed)
