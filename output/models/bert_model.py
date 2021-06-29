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
                          AutoModelForSequenceClassification, get_cosine_schedule_with_warmup)
from transformers import (RobertaTokenizer, RobertaModel)
from torch.utils.tensorboard import SummaryWriter


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(AttentionHead, self).__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim

        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

        self.init_weights(self.W, self.V)

    def init_weights(self, *blocks):
        for m in blocks:
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
        self.head = AttentionHead(768,768)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.head.out_features, 1)

        self.init_weights(self.linear)

    def init_weights(self, *blocks):
        for m in blocks:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, **xb):
        x = self.roberta(**xb)[0]
        x = self.head(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class BertTrainer:
    def __init__(self, args, train_loader, test_loader):
        self.__dict__ = args
        self.model = Model().to(args.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.01)
        # self.lr_scheduler = get_cosine_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=0, num_training_steps= 10 * len(train_loader))
        self.lr_scheduler = None

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.writer = SummaryWriter(
            log_dir=os.path.join(self.logdir) if self.logdir != '' else None)

    def resume_training(self):
        if os.path.isfile(self.resume):
            print("=> loading checkpoint '{}'".format(self.resume))
            checkpoint = torch.load(self.resume)
            self.start_epoch = checkpoint['epoch']
            self.best_test_loss = checkpoint['best_test_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.resume, checkpoint['epoch']))

    def save_checkpoint(self, epoch, model_dir, filename='checkpoint.pth.tar'):
        save_dir = os.path.join(self.checkpoint_dir, model_dir)
        state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_test_loss': self.best_test_loss,
                'optimizer': self.optimizer.state_dict(),
            }
        torch.save(state, os.path.join(save_dir, filename))

    def loss_fn(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        return torch.sqrt(nn.MSELoss()(outputs, targets))

    def train(self):
        with tqdm(desc="Epoch", total=self.epochs) as progress:
            for epoch in range(self.epochs):
                print (f"[Epoch {epoch}] started")
                self.train_epoch(epoch)

                progress.update(1)

    def adjust_lr(self, i):
        lr = self.lr * (self.lr_coef ** ((i + 1) // self.lr_interval))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_epoch(self, epoch):
        self.model.train()
        train_losses = []
        best_test_loss = 999999
        with tqdm(desc="Batch", total=len(self.train_loader)) as progress:
            for i, (X_batch, y_batch) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                y_batch = torch.Tensor(y_batch.float()).to(self.device)
                inputs = {key: val.reshape(val.shape[0], -1).to(self.device) for key, val in X_batch.items()}

                y_pred = self.model(**inputs)
                # print(y_pred.detach().cpu().numpy(), y_batch)
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                else:
                    self.adjust_lr(i)

                train_losses.append(loss.item())

                # Logging
                counter = epoch * len(self.train_loader) + i
                if ((i+1) % self.eval_step == 0) or ((i + 1) == len(self.train_loader)):
                    print(f"[Epoch {epoch}] Train loss: ", np.mean(train_losses), '\n')
                    eval_loss = self.evaluate(epoch)
                    self.writer.add_scalar("Train loss", np.mean(train_losses), counter)
                    self.writer.add_scalar("Test loss", eval_loss, counter)
                    if eval_loss < best_test_loss:
                        self.save_checkpoint(epoch, model_dir=self.model_name + str(self.fold))
                progress.update(1)

    def evaluate(self, epoch):
        test_losses = []
        self.model.eval()
        with torch.no_grad():
            with tqdm(desc="Batch", total=len(self.test_loader)) as progress:
                for i,(X_batch, y_batch) in enumerate(self.test_loader):

                    y_batch = torch.Tensor(y_batch.float()).to(self.device)

                    inputs = {key: val.reshape(val.shape[0], -1).to(self.device) for key, val in X_batch.items()}

                    y_pred = self.model(**inputs)
                    loss = self.loss_fn(y_pred, y_batch)
                    test_losses.append(loss.item())

                    progress.update(1)
        print(f"[Epoch {epoch}] Validation loss: ", np.mean(test_losses), '\n')
        return np.mean(test_losses)


def run(args, train_loader, test_loader, verbose=True):
    def loss_fn(outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        return torch.sqrt(nn.MSELoss()(outputs, targets))

    def train_and_evaluate_loop(train_loader, valid_loader, model, loss_fn, optimizer, epoch, best_loss,
                                valid_step=10, lr_scheduler=None):
        train_loss = 0
        for i, (inputs1, targets1) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            inputs1 = {key: val.reshape(val.shape[0], -1).to(device) for key, val in inputs1.items()}
            outputs1 = model(**inputs1)
            loss1 = loss_fn(outputs1, targets1.to(device))
            loss1.backward()
            optimizer.step()

            train_loss += loss1.item()

            if lr_scheduler:
                lr_scheduler.step()

            # evaluating for every valid_step
            if (i % valid_step == 0) or ((i + 1) == len(train_loader)):
                model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for j, (inputs2, targets2) in enumerate(valid_loader):
                        inputs2 = {key: val.reshape(val.shape[0], -1).to(device) for key, val in inputs2.items()}
                        outputs2 = model(**inputs2)
                        loss2 = loss_fn(outputs2, targets2.to(device))
                        valid_loss += loss2.item()

                    valid_loss /= len(valid_loader)
                    if valid_loss <= best_loss:
                        if verbose:
                            print(f"epoch:{epoch} | Train Loss:{train_loss / (i + 1)} | Validation loss:{valid_loss}")

                        best_loss = valid_loss

        return best_loss

    device = args.device

    model = Model()

    train_dl = train_loader

    valid_dl = test_loader

    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=10 * len(train_dl))

    model = model.to(device)

    best_loss = 9999
    for epoch in range(args["num_epochs"]):
        print(f"Epoch Started:{epoch}")
        best_loss = train_and_evaluate_loop(train_dl, valid_dl, model, loss_fn,
                                            optimizer, epoch, best_loss,
                                            valid_step=10, lr_scheduler=lr_scheduler)