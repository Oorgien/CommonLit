import random
import warnings
warnings.filterwarnings('ignore')

import torch

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

    args = dotdict({
        'collab': False,
        'seed': 42,

        'logdir': 'runs',
        'checkpoint_dir': 'checkpoints',
        'model_name': 'Bert_adamW',

        'norm': False,
        'nfolds': 5,
        'batch_size': 14,
        'epochs': 6,
        'eval_step': 50,
        'max_len': 256,

        'lr': 2e-5,
        'lr_coef': 0.5,
        'lr_interval': 50,

        'resume': '',
        'train_data_path': '../input/commonlitreadabilityprize/train.csv',
        'test_data_path': '../input/commonlitreadabilityprize/test.csv',
        'sample_path': '../input/commonlitreadabilityprize/sample_submission.csv'
    })

    if not os.path.isdir(os.path.join(args.checkpoint_dir, args.model_name)):
        os.makedirs(os.path.join(args.checkpoint_dir, args.model_name))
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=args.seed)

    # Get train data
    train_data = pd.read_csv(args.train_data_path)
    args.train_data = train_data

    num_bins = int(np.floor(1 + np.log2(len(train_data))))
    train_data.loc[:, 'bins'] = pd.cut(train_data['target'], bins=num_bins, labels=False)

    bins = train_data.bins.to_numpy()
    target = train_data.target.to_numpy()

    kfold = StratifiedKFold(n_splits=args.nfolds, shuffle=True, random_state=args.seed)
    for k, (train_idx, valid_idx) in enumerate(kfold.split(X=train_data, y=bins)):
        train_data.loc[valid_idx, 'Fold'] = k
        args.fold = k
        print (f"Training on fold {k}")

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

        bert_trainer = BertTrainer(args, train_loader, test_loader)
        bert_trainer.train()
