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


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_targets):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim

        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        att = torch.tanh(self.W(features))

        score = self.V(att)

        attention_weights = torch.softmax(score, dim=1)

        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.head = AttentionHead(768,768,1)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.head.out_features,1)

        self.init_weights(self.linear)

    def init_weights(self, *blocks):
        for m in blocks:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,**xb):
        x = self.roberta(**xb)[0]
        x = self.head(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class BertTrainer:
    def __init__(self, config, train_loader, test_loader):
        self.model = Model().to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.epochs = config.num_epochs
        self.device = config.device

    def train(self):
        with tqdm(desc="Epoch", total=self.epochs) as progress:
            for epoch in tqdm(self.epochs):
                self.train_epoch()
                self.eval_epoch()

                progress.update(1)

    def train_epoch(self):
        with tqdm(desc="Batch", total=len(self.train_loader)) as progress:
            for ids, X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()

                y_batch = torch.Tensor(y_batch.float()).to(self.device)
                inputs = {key: val.reshape(val.shape[0], -1).to(self.device) for key, val in X_batch.items()}

                y_pred = self.model.forward(**inputs)

                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()

                self.optimizer.step()
                progress.update(1)

    def eval_epoch(self):
        with tqdm(desc="Batch", total=len(self.test_loader)) as progress:
            for ids, X_batch, y_batch in self.test_loader:

                y_batch = torch.Tensor(y_batch.float()).to(self.device)

                inputs = {key: val.reshape(val.shape[0], -1).to(self.device) for key, val in X_batch.items()}

                y_pred = self.model(**inputs)
                loss = self.loss_fn(y_pred, y_batch)

                print(loss)
                progress.update(1)