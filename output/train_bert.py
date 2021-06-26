import os
import gc
import sys
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
from prepare_data import *
from models.bert_model import BertTrainer, run

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

        'batch_size': 14,
        'epochs': 10,
        'eval_step': 50,
        'logdir': 'runs',

        'lr': 2e-5,
        'lr_coef': 0.5,
        'lr_interval': 50,

        'max_len': 256,
        'train_data_path': '../input/commonlitreadabilityprize/train.csv',
        'test_data_path': '../input/commonlitreadabilityprize/test.csv',
        'sample_path': '../input/commonlitreadabilityprize/sample_submission.csv'
    })

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=config.seed)

    train_loader, test_loader = init_loaders(config)

    bert_trainer = BertTrainer(config, train_loader, test_loader)
    bert_trainer.train()

    run(config, train_loader, test_loader)
