import time
import datetime
import random
import sys
import os
import yaml
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
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
import sklearn
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt 

current_working_directory = os.getcwd()
print('current working directory is', current_working_directory)
sys.path.append(current_working_directory) 

from abcdfusion import metrics
from abcdfusion.models import BinaryMLP, LinearClassifier, Ensemble
from abcdfusion.utils import check_make_dir
from abcdfusion.preprocess import get_preprocess
from abcdfusion import get_abcd, get_cv_splits, create_datasets, train_epoch_multi, test_multi

"""
example command to run the script:
python examples/multi_modalities_cv.py -c ./configs.yaml -e ./exps/multi -n 0 -m DTI RS CORT OTHER -l bce -s 42 -g -1 -f 0 -ne 10 -op 'SGD' -lr 0.0001 -do 0.0 -b -d
"""

parser = argparse.ArgumentParser(description='Model Cross Validation')
parser.add_argument('--config', '-c', type=str, required=True,
                    help='config file that defines loading data')
parser.add_argument('--experiment', '-e', type=str, required=True,
                    help='name of the experiment')
parser.add_argument('--name', '-n', type=str, required=True, 
                    help='name of run', )
parser.add_argument('--modalities', '-m', choices=['DTI', 'RS', 'OTHER', 'CORT'], 
                    nargs='+', required=True, help='which modalities to use')
parser.add_argument('--loss', '-l', type=str, default='bce',
                    help='the loss function to use')
parser.add_argument('--seed', '-s', type=int, default=None, 
                    help='which seed for random number generator to use')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='which GPU to use, negative value denotes cpu will be used')
parser.add_argument('--frequency', '-f', type=int, default=0,
                    help='for every # epochs to save checkpoints')
parser.add_argument('--num_epochs', '-ne', type=int, default=100,
                    help='the number of epochs for training')
parser.add_argument('--optmizer', '-op', type=str, default='SGD',
                    help='the optimizer for training')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001,
                    help='the learning rate for training')
parser.add_argument('--drop_out', '-do', type=float, default=0.0,
                    help='the drop out rate for training')
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

# set up benchmark running
if args.benchmark:
    torch.backends.cudnn.benchmark = True
else:
    torch.backends.cudnn.benchmark = False
    
if args.debug:
    torch.autograd.set_detect_anomaly(True)
else:
    torch.autograd.set_detect_anomaly(False)
    sys.stdout = open(os.path.join(log_path, 'log.txt'), 'w')
    
modalities = [modality.upper() for modality in args.modalities]
optmizer_name = args.optmizer
lr = float(args.learning_rate)
p = float(args.drop_out)
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
    
writer = SummaryWriter(log_path)

# loading data
auxiliary = configs['Auxiliary']
DATA_PATH = auxiliary['DATA_PATH']

dti_file = os.path.join(DATA_PATH, auxiliary['DTI_DATA'])
rs_file = os.path.join(DATA_PATH, auxiliary['RS_DATA'])
cort_file = os.path.join(DATA_PATH, auxiliary['COR_DATA'])
other_file = os.path.join(DATA_PATH, auxiliary['OTHER_DATA'])
outcome_file = os.path.join(DATA_PATH, auxiliary['OUTCOME'])

desired_metrics = configs['METRICS']

modality2index = {}
index2modality = {}
n_modality = 0
if 'DTI' in modalities:
    dti_model_params = configs['DTI']
    modality2index['DTI'] = n_modality
    index2modality[n_modality] = 'DTI'
    n_modality += 1
if 'RS' in modalities:
    rs_model_params = configs['RS']
    modality2index['RS'] = n_modality
    index2modality[n_modality] = 'RS'
    n_modality += 1
if 'CORT' in modalities:
    cort_model_params = configs['CORT']
    modality2index['CORT'] = n_modality
    index2modality[n_modality] = 'CORT'
    n_modality += 1
if 'OTHER' in modalities:
    other_model_params = configs['OTHER']
    modality2index['OTHER'] = n_modality
    index2modality[n_modality] = 'OTHER'
    n_modality += 1

metrics_dict = {}
arrays = get_abcd(dti_file, rs_file, other_file, cort_file, outcome_file)
X, y, groups = get_preprocess(arrays, modalities, group_site=True, combine=True)
splitter = get_cv_splits(n_splits=0, group_site=True)

for i, (train_index, test_index) in enumerate(splitter.split(X[0], y, groups=groups)):
    fold_groups = ','.join([str(int(group)) for group in set(groups[test_index])])
    print(f"Fold {i}; Test Sites {fold_groups}; There are {len(train_index)} training subjects and {len(test_index)} testing subjects.")
    input_sizes = {}
    for index, mod in index2modality.items():
        input_sizes[mod] = len(X[index][0])
    
    feature_extractors = []
    output_sizes = {}
    if dti_model_params:
        if dti_model_params['architecture'] == 'mlp':
            dti_model = BinaryMLP(input_sizes['DTI'], dti_model_params['layers'])
            out = dti_model(torch.rand(1, input_sizes['DTI']))
            output_sizes['DTI'] = len(out[0])
            feature_extractors.append(dti_model)
    if rs_model_params:
        if rs_model_params['architecture'] == 'mlp':
            rs_model = BinaryMLP(input_sizes['RS'], rs_model_params['layers'])
            out = rs_model(torch.rand(1, input_sizes['RS']))
            output_sizes['RS'] = len(out[0])
            feature_extractors.append(rs_model)
    if cort_model_params:
        if cort_model_params['architecture'] == 'mlp':
            cort_model = BinaryMLP(input_sizes['CORT'], cort_model_params['layers'])
            out = cort_model(torch.rand(1, input_sizes['CORT']))
            output_sizes['CORT'] = len(out[0])
            feature_extractors.append(cort_model)
    if other_model_params:
        if other_model_params['architecture'] == 'mlp':
            other_model = BinaryMLP(input_sizes['OTHER'], other_model_params['layers'])
            out = other_model(torch.rand(1, input_sizes['OTHER']))
            output_sizes['OTHER'] = len(out[0])
            feature_extractors.append(other_model)
    discriminator = LinearClassifier(sum(output_sizes.values()), p=p)
    model = Ensemble(discriminator, *feature_extractors)
            
    examplar = []
    for mod, size in input_sizes.items():
        examplar.append(torch.rand(1, size))
    out = model(examplar)
    print('output shape', out.shape) 
    model = model.to(device)
        
    X_train = [X_s[train_index] for X_s in X]
    y_train = y[train_index]
    X_test = [X_s[test_index] for X_s in X]
    y_test = y[test_index]
    groups_train = groups[train_index]
    groups_test = groups[test_index]
    train_loader = create_datasets(X_train, y_train)
    test_loader = create_datasets(X_test, y_test)
    
    optimizer = getattr(optim, optmizer_name)(model.parameters(), lr=lr)
    start_epoch = 0
    writer.add_text(f'Fold_{i}-sites', fold_groups, start_epoch)

    print('Starting training loop; initial compile can take a while...')
    since = time.time()
    model.train()   # Set model to evaluate mode

    pbar = trange(num_epochs, desc='Epoch', unit='epoch', initial=start_epoch, position=0, disable=not args.debug)
    # Iterate over data.
    for epoch in pbar:
        model, train_stats = train_epoch_multi(model, train_loader, criterion, optimizer, device)

        if writer:
            writer.add_scalar('time eplased', time.time() - since, epoch)
            for stat in train_stats:
                writer.add_scalar(stat, train_stats[stat], epoch)
            if epoch+1==num_epochs or (frequency>0 and epoch%frequency==0):
                preds, test_stats = test_multi(model, test_loader, device)
                # compute group level confusion matrix
                cfms = metrics.get_group_confusion_matrix(preds.squeeze(), y_test.squeeze(), groups=groups_test)
                metrics_dict.update(metrics.compute_metrics(cfms, desired_metrics))
                for stat in test_stats:
                    writer.add_scalar(f'Fold_{i}-'+stat, test_stats[stat], epoch)

        pbar.set_postfix(loss = train_stats['train_loss'], acc = train_stats['train_acc'])

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
df = pd.DataFrame.from_dict(metrics_dict)
df.to_csv(os.path.join(log_path, 'results.csv'), index=False)

writer.flush()
writer.close()