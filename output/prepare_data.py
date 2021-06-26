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
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

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


def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def prepare_data(config):
    train_data = pd.read_csv(config.train_data_path)
    test_data = pd.read_csv(config.test_data_path)
    sample = pd.read_csv(config.sample_path)

    config.train_data = train_data
    config.test_data = test_data
    config.sample = sample

    target = train_data['target'].to_numpy()
    target_normed = (target - np.mean(target)) / np.var(target)
    config.target_mean = np.mean(target)
    config.target_var = np.var(target)
    train_data['target_normed'] = target_normed

    X_train, X_test, y_train, y_test = train_test_split(
        train_data,
        pd.DataFrame(train_data['target_normed']), test_size=0.33, random_state=config.seed)

    return X_train, X_test, \
           y_train.rename(columns={"target_normed": "target"}), \
           y_test.rename(columns={"target_normed": "target"})


def init_loaders(config):
    X_train, X_test, y_train, y_test = prepare_data(config)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train_dataset = CLRPDataset(X_train, y_train, tokenizer, config)
    test_dataset = CLRPDataset(X_test, y_test, tokenizer, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader


class CLRPDataset(Dataset):
    def __init__(self, X, y, tokenizer, config):
        self.excerpt = X['excerpt'].to_numpy()
        self.targets = y['target'].to_numpy()
        self.max_len = config.max_len
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        encode = self.tokenizer(self.excerpt[idx],
                                return_tensors='pt',
                                max_length=self.max_len,
                                padding='max_length',
                                truncation=True)

        target = torch.tensor(self.targets[idx], dtype=torch.float)
        return encode, target

    def __len__(self):
        return len(self.excerpt)

