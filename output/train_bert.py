import random
import warnings
warnings.filterwarnings('ignore')

import torch

from utils import dotdict
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, StratifiedShuffleSplit
from prepare_data import init_loaders
from models.bert_model import BertTrainer, run, Model

import os

import random
import numpy as np
import pandas as pd


import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    args = dotdict({
        'collab': False,
        'seed': 42,

        'logdir': 'runs',
        'checkpoint_dir': 'checkpoints',
        'model_name': 'Bert_adamW',
        'model_log_name': '4layes_lr2e-5',

        'norm': False,
        'nfolds': 5,
        'batch_size': 16,
        'epochs': 10,
        'max_len': 256,
        'valid_step': 10,

        'lr': 2e-5,
        'warmup': 0,
        'lr_coef': 0.99,
        'lr_interval': 5,
        'lr_change': 'scheduler',

        'resume': '',
        'train_data_path': '../input/commonlitreadabilityprize/train.csv',
        'test_data_path': '../input/commonlitreadabilityprize/test.csv',
        'sample_path': '../input/commonlitreadabilityprize/sample_submission.csv'
    })

    # Checkpoints
    args.save_dir = os.path.join(
            args.checkpoint_dir, os.path.join(args.model_name, args.model_log_name))
    if not os.path.isdir(os.path.join(
            args.checkpoint_dir, os.path.join(args.model_name, args.model_log_name))):
        os.makedirs(os.path.join(
            args.checkpoint_dir, os.path.join(args.model_name, args.model_log_name)))

    # Summary writer
    args.summary_dir = os.path.join(args.logdir, args.model_log_name)
    if not os.path.isdir(os.path.join(args.logdir, args.model_log_name)):
        os.makedirs(os.path.join(args.logdir, args.model_log_name))

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (args.device)
    seed_everything(seed=args.seed)

    # Get train data
    train_data = pd.read_csv(args.train_data_path)
    args.train_data = train_data

    num_bins = int(np.floor(1 + np.log2(len(train_data))))
    train_data.loc[:, 'bins'] = pd.cut(train_data['target'], bins=num_bins, labels=False)

    bins = train_data.bins.to_numpy()

    kfold = StratifiedKFold(n_splits=args.nfolds, shuffle=True, random_state=args.seed)
    train_data['Fold'] = -1
    for k, (train_idx, valid_idx) in enumerate(kfold.split(X=train_data, y=bins)):
        train_data.loc[valid_idx, 'Fold'] = k
        args.fold = k
        print (f"Training on fold {k}")

        # X_train, X_val = train_data.query(f"Fold != {k}"), train_data.query(f"Fold == {k}")
        # y_train, y_val = X_train['target'], X_val['target']
        X_train, X_val = train_data.iloc[train_idx, :], train_data.iloc[valid_idx, :]
        y_train, y_val = train_data['target'].iloc[train_idx], train_data['target'].iloc[valid_idx]

        if args.norm:
            y_train = (y_train - np.mean(y_train)) / np.var(y_train)
            y_val = (y_val - np.mean(y_train)) / np.var(y_train)

            args.target_mean = np.mean(y_train)
            args.target_var = np.var(y_train)

        train_loader, test_loader = init_loaders(
            args, X_train, X_val,
            pd.DataFrame(y_train, columns=['target']),
            pd.DataFrame(y_val, columns=['target']))
        # run(args, train_loader, test_loader)

        model = Model().to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        bert_trainer = BertTrainer(args, model, optimizer, train_loader, test_loader)
        bert_trainer.train()
