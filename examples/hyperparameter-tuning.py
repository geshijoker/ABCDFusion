import time
import datetime
import random
import sys
import os
import yaml
import argparse
from functools import partial

import numpy as np
from tqdm import tqdm, trange
from ptflops import get_model_complexity_info
from torchinfo import summary

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt 

current_working_directory = os.getcwd()
print('current working directory is', current_working_directory)
sys.path.append(current_working_directory) 

from abcdfusion import metrics
from abcdfusion.models import BinaryMLP, LinearClassifier
from abcdfusion.utils import check_make_dir
from abcdfusion.preprocess import get_preprocess
from abcdfusion import get_abcd, get_cv_splits, create_datasets, train_epoch_single, test_single

"""
This is the script for hyper-parameter tuning using ray tune.
example command to run the script:
python examples/hyperparameter-tuning.py -c ./configs.yaml -m DTI -l bce -s 42 -f 0 -ne 10 -op 'SGD'
"""

parser = argparse.ArgumentParser(description='Training hyperparameter tuning')
parser.add_argument('--config', '-c', type=str, required=True,
                    help='config file that defines loading data')
parser.add_argument('--modality', '-m', choices=['DTI', 'RS', 'OTHER', 'CORT'], required=True,
                    help='which modality to use')
parser.add_argument('--loss', '-l', type=str, default='bce',
                    help='the loss function to use')
parser.add_argument('--seed', '-s', type=int, default=None, 
                    help='which seed for random number generator to use')
parser.add_argument('--frequency', '-f', type=int, default=0,
                    help='for every # epochs to save checkpoints')
parser.add_argument('--num_epochs', '-ne', type=int, default=100,
                    help='the number of epochs for training')
parser.add_argument('--optmizer', '-op', type=str, default='SGD',
                    help='the optimizer for training')
parser.add_argument('--benchmark', '-b', action='store_true',
                    help='using benchmark algorithms')
parser.add_argument('--debug', '-d', action='store_true',
                    help='using debug mode')

# load and parse argument
args = parser.parse_args()

with open(args.config, 'r') as file:
    try:
        configs = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        sys.exit(exc)

# set up the seed
if args.seed:
    seed = args.seed
else:
    seed = torch.seed()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

modality = args.modality.upper()
optmizer_name = args.optmizer
frequency = args.frequency
num_epochs = args.num_epochs
classes = (0, 1)
n_classes = len(classes)

if args.loss == 'bce':
    print('use Binary CrossEntropy with logits')
    pos_weight = torch.tensor([20.0])  # number of instances in negative class / number of instances in positive class
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    print(f'{args.loss} not supported, use default CrossEntropyLoss')
    pos_weight = torch.tensor([20.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
# loading data
auxiliary = configs['Auxiliary']
DATA_PATH = auxiliary['DATA_PATH']

dti_file = os.path.join(DATA_PATH, auxiliary['DTI_DATA'])
rs_file = os.path.join(DATA_PATH, auxiliary['RS_DATA'])
cort_file = os.path.join(DATA_PATH, auxiliary['COR_DATA'])
other_file = os.path.join(DATA_PATH, auxiliary['OTHER_DATA'])
outcome_file = os.path.join(DATA_PATH, auxiliary['OUTCOME'])

desired_metrics = configs['METRICS']
model_params = configs[modality]

arrays = get_abcd(dti_file, rs_file, other_file, cort_file, outcome_file)
X, y, groups = get_preprocess(arrays, [modality], group_site=True, combine=True)

X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(X[0], y.squeeze(), groups, test_size=0.2, random_state=seed)

input_size = len(X_train[0])
    
train_loader = create_datasets([X_train], y_train)
test_loader = create_datasets([X_test], y_test)

def train_tune(config, input_size):
    if model_params['architecture'] == 'mlp':
        model = nn.Sequential(
            BinaryMLP(input_size, model_params['layers']*config['size']),
            LinearClassifier(model_params['layers'][-1]*config['size'], p=config['p'])
        )
    out = model(torch.rand(1, input_size))
    print('output shape', out.shape) 
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = getattr(optim, optmizer_name)(model.parameters(), lr=config['lr'])
    
    print('Starting training loop; initial compile can take a while...')
    since = time.time()
    model.train()   # Set model to evaluate mode
    start_epoch = 0

    pbar = trange(num_epochs, desc='Epoch', unit='epoch', initial=start_epoch, position=0, disable=True)
    # Iterate over data.
    for epoch in pbar:
        model, train_stats = train_epoch_single(model, train_loader, criterion, optimizer, device)
        preds, test_stats = test_single(model, test_loader, device)
        session.report(
            {"loss": train_stats['train_loss'], "accuracy": test_stats['test_acc']},
        )
    

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    config = {
        "size": tune.choice([1,2,3]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "p": tune.uniform(0.0, 0.2)
    }
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_tune, input_size=input_size),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final training loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=1000, gpus_per_trial=0)