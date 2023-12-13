import time
import datetime
import random
import sys
import os
import argparse

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
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

import pandas as pd
import numpy as np
import sklearn
import matplotlib
from matplotlib import pyplot as plt 

from abcdfusion import metrics
from abcdfusion.models import BinaryMLP
from abcdfusion.utils import check_make_dir
from abcdfusion import get_abcd, train_epoch_single, test_single, preprocess

"""
python examples/single_modality_train.py -c ./config.yaml -e ./exps -n 0 -id 0 -ls 32 64 32 -a mlp -l ce -s 42 -g -1 -f 1 -ne 10 -lr 0.0001 -bs 1000 -t -b -d
"""

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--config', '-c', type=str, required=True,
                    help='config file that defines loading data')
parser.add_argument('--experiment', '-e', type=str, required=True,
                    help='name of the experiment')
parser.add_argument('--name', '-n', type=str, required=True, 
                    help='name of run', )
parser.add_argument('--index', '-id', type=int, required=True,
                    help='which modality to use')
parser.add_argument('--loss', '-l', type=str, default='bce',
                    help='the loss function to use')
parser.add_argument('--resume', '-r', type=str, default=None, 
                    help='resume from checkpoint')
parser.add_argument('--seed', '-s', type=int, default=None, 
                    help='which seed for random number generator to use')
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='which GPU to use, only when disable-cuda not specified')
parser.add_argument('--frequency', '-f', type=int, default=0,
                    help='for every # epochs to save checkpoints')
parser.add_argument('--num_epochs', '-ne', type=int, default=100,
                    help='the number of epochs for training')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001,
                    help='the learning rate of training')
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

if args.gpu<0 or not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    if args.gpu<torch.cuda.device_count():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device("cuda") 
print('Using device: {}'.format(device))

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
    
experiment = args.experiment
run_name = args.name + f'_seed_{seed}'
log_path = os.path.join(experiment, run_name)

if os.path.isdir(log_path):
    sys.exit('The name of the run has alrealy exist')
else:
    check_make_dir(log_path)
sys.stdout = open(os.path.join(log_path, 'log.txt'), 'w')

# set up benchmark running
if args.benchmark:
    torch.backends.cudnn.benchmark = True
else:
    torch.backends.cudnn.benchmark = False
    
if args.debug:
    torch.autograd.set_detect_anomaly(True)
else
    torch.autograd.set_detect_anomaly(False)
    
index = args.index
layer_sizes = args.layer_sizes
lr = args.learning_rate
frequency = args.frequency
test_while_train = args.test_while_train
num_epochs = args.num_epochs
    
# writer = SummaryWriter(log_path)

# loading data
DATA_PATH = configs['DATA_PATH']

dti_file = os.path.join(DATA_PATH, configs['DTI_DATA'])
rsfmri_file = os.path.join(DATA_PATH, configs['RS_DATA'])
other_file = os.path.join(DATA_PATH, configs['OTHER_DATA'])
cort_file = os.path.join(DATA_PATH, configs['COR_DATA'])
outcome_file = os.path.join(DATA_PATH, configs['OUTCOME'])

abcd_dataset = get_abcd(dti_file, rsfmri_file, other_file, cort_file, outcome_file)
train_loader, valid_loader = create_datasets(abcd_dataset, batch_size, 0, 0.2, shuffle=True)
data = next(iter(valid_loader))
print('modality size: ', data[index].shape, data[-1].shape)

input_size = data[index].shape[-1]
x = torch.rand(1, input_size)

if args.architecture == 'mlp':
    model = BinaryMLP(input_size, layer_sizes, p=0.2)
out = model(x)
print('output shape', out.shape) 

if args.loss == 'bce':
    print('use Binary CrossEntropy with logits')
    criterion = nn.BCEWithLogitsLoss(pos_weight=10)
else:
    print(f'{args.loss} not supported, use default CrossEntropyLoss')
    criterion = nn.BCEWithLogitsLoss(pos_weight=10)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20)

start_epoch = 0

if args.resume:
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    

def save_checkpoint(checkpoint):
    utctime = datetime.datetime.now(datetime.timezone.utc).strftime("%m-%d-%Y-%H:%M:%S")
    model_path = os.path.join(log_path, f'iter{checkpoint}-' + utctime + '.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': checkpoint,
        }, model_path)

print('Starting training loop; initial compile can take a while...')
since = time.time()
model.train()   # Set model to evaluate mode

pbar = trange(num_epochs, desc='Epoch', unit='epoch', initial=start_epoch, position=0)
# Iterate over data.
for epoch in pbar:
    model, train_stats = train_epoch_single(model, train_loader, index, n_classes, criterion, optimizer, scheduler, device)
    if test_while_train:
        test_stats = test_single(model, test_loader, index, n_classes, device)

    if writer:
        writer.add_scalar('time eplased', time.time() - since, epoch)
        for stat in train_stats:
            writer.add_scalar(stat, train_stats[stat], epoch)
        if epoch+1==num_epochs or (frequency>0 and epoch%frequency==0):
            for stat in test_stats:
                writer.add_scalar(stat, test_stats[stat], epoch)
            save_checkpoint(epoch)

    pbar.set_postfix(loss = train_stats['epoch_loss'], acc = train_stats['epoch_acc'])

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s)

writer.flush()
writer.close()