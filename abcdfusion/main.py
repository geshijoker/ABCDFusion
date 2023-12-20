import csv
import yaml
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut, LeaveOneOut

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torcheval.metrics.aggregation.auc import AUC

from abcdfusion import metrics
from abcdfusion import models
from abcdfusion import utils

def get_abcd(DTI_csv, rsfMRI_csv, other_csv, cort_csv, label_csv):
    """
    Load CSV files into numpy arrays
    """
    df_dti = pd.read_csv(DTI_csv, index_col=0)
    df_rs = pd.read_csv(rsfMRI_csv, index_col=0)
    df_other = pd.read_csv(other_csv, index_col=0)
    df_cort = pd.read_csv(cort_csv, index_col=0)
    df_labels = pd.read_csv(label_csv, index_col=0)

    X_dti = df_dti.iloc[:,:].values
    X_rs = df_rs.iloc[:,:].values
    X_other = df_other.iloc[:,:].values
    X_cort = df_cort.iloc[:,:].values
    X_anno = df_labels.iloc[:,:].values

    return [X_dti, X_rs, X_other, X_cort, X_anno]

def get_cv_splits(n_splits=0, group_site=True):
    """
    Retrieve cross validation 
    """
    # groups: site
    if n_splits<=0 and group_site:
        splitter = LeaveOneGroupOut()
    elif n_splits>0 and group_site:
        splitter = GroupKFold(n_splits=n_splits)
    elif n_splits<=0 and not group_site:
        splitter = LeaveOneOut()
    else:
        splitter = KFold(n_splits=n_splits)
    return splitter

def create_datasets(arrays, y):
    """
    numpy arrays to tensors
    """
    # obtain training indices that will be used for validation
    # groups: age, sex, site
    
    arrays = [torch.Tensor(array) for array in arrays]
    dataset = TensorDataset(*arrays, torch.Tensor(y))
    num_data = len(dataset)
    dataloader = DataLoader(dataset, batch_size=num_data, shuffle=False)
    
    return dataloader

def train_epoch_single(model, dataloader, criterion, optimizer, device):
    """
    Train a model with single modality
    Returns: 
        model: pytorch model
        train_stats: dictionary of training stats such as loss
    """
    epoch_loss = 0.0
    epoch_acc = 0.0
    count = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)
        nxt_count = count+batch_size
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        preds = torch.round(outputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # statistics
        epoch_loss = loss.item() * batch_size/nxt_count + epoch_loss * count/nxt_count
        epoch_acc = ((preds == targets).sum()/np.prod(preds.size())).item() * batch_size/nxt_count + epoch_acc * count/nxt_count
        
        count = nxt_count

    train_stats = {
        'train_loss': epoch_loss,
        'train_acc': 100. * epoch_acc,
    }
    
    return model, train_stats

def test_single(model, dataloader, device):
    """
    Test a model with single modality
    Returns:
        predictions: numpy array
        test_stats: dictionary of test stats such as accuracy
    """
    since = time.time()
    model.eval()   # Set model to evaluate mode
    
    corrects = 0
    count = 0

    # Iterate over data.
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            batch_size = inputs.size(0)
            count += batch_size

            outputs = model(inputs)
            preds = torch.round(outputs)

            # statistics
            corrects += torch.sum(preds == targets.data)/np.prod(preds.size())*batch_size

    acc = corrects.double().item() / count

    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, Test Acc: {100. * acc}')
    
    test_stats = {
        "test_acc": 100. * acc,
    }

    return preds.detach().cpu().numpy(), test_stats

def train_epoch_multi(model, dataloader, criterion, optimizer, device):
    """
    Train an emsemble model with multiple modalities
    Returns: 
        model: pytorch model
        train_stats: dictionary of training stats such as loss
    """
    epoch_loss = 0.0
    epoch_acc = 0.0
    count = 0

    for data in dataloader:
        inputs = [inp.to(device) for inp in data[:-1]]
        targets = data[-1].to(device)

        batch_size = inputs[0].size(0)
        nxt_count = count+batch_size
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        preds = torch.round(outputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # statistics
        epoch_loss = loss.item() * batch_size/nxt_count + epoch_loss * count/nxt_count
        epoch_acc = ((preds == targets).sum()/np.prod(preds.size())).item() * batch_size/nxt_count + epoch_acc * count/nxt_count
        
        count = nxt_count

    train_stats = {
        'train_loss': epoch_loss,
        'train_acc': 100. * epoch_acc,
    }
    
    return model, train_stats

def test_multi(model, dataloader, device):
    """
    Test an emsemble model with multiple modalities
    Returns:
        predictions: numpy array
        test_stats: dictionary of test stats such as accuracy
    """
    since = time.time()
    model.eval()   # Set model to evaluate mode
    
    corrects = 0
    count = 0

    # Iterate over data.
    with torch.no_grad():
        for data in dataloader:
            inputs = [inp.to(device) for inp in data[:-1]]
            targets = data[-1].to(device)
            
            batch_size = inputs[0].size(0)
            count += batch_size

            outputs = model(inputs)
            preds = torch.round(outputs)

            # statistics
            corrects += torch.sum(preds == targets.data)/np.prod(preds.size())*batch_size

    acc = corrects.double().item() / count
    
    metric = AUC()
    metric.update(preds.data, targets.data)
    auc = metric.compute()

    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, Test Acc: {100. * acc}')
    
    test_stats = {
        "test_acc": 100. * acc,
        "test_auc": auc[0],
    }

    return preds.detach().cpu().numpy(), test_stats
        
if __name__=='__main__':
    config_file = 'configs.yaml'
    with open(config_file, 'r') as infile:
        try:
            configs = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            sys.exit(exc)
            
    auxiliary = configs['Auxiliary']
    DATA_PATH = auxiliary['DATA_PATH']
    dti_file = os.path.join(DATA_PATH, auxiliary['DTI_DATA'])
    rsfmri_file = os.path.join(DATA_PATH, auxiliary['RS_DATA'])
    other_file = os.path.join(DATA_PATH, auxiliary['OTHER_DATA'])
    cort_file = os.path.join(DATA_PATH, auxiliary['COR_DATA'])
    outcome_file = os.path.join(DATA_PATH, auxiliary['OUTCOME'])
    
    dti, rs, other, cort, y = get_abcd(dti_file, rsfmri_file, other_file, cort_file, outcome_file)
    print(dti.shape, rs.shape, other.shape, cort.shape, y.shape)