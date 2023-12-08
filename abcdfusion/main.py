import csv
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, TensorDataset, DataLoader

from abcdfusion import metrics
from tqdm import tqdm, trange

def get_abcd(DTI_csv, rsfMRI_csv, other_csv, cort_csv, diff_csv, label_csv):
    df_dti = pd.read_csv(DTI_csv)
    df_rs = pd.read_csv(rsfMRI_csv)
    df_other = pd.read_csv(other_csv)
    df_cort = pd.read_csv(cort_csv)
    df_diff = pd.read_csv(diff_csv)
    df_labels = pd.read_csv(label_csv)

    X_dti = torch.Tensor(df_dti.iloc[:,3:].values)
    X_rs = torch.Tensor(df_rs.iloc[:,2:].values)
    X_other = torch.Tensor(df_other.iloc[:,1:].values)
    X_cort = torch.Tensor(df_cort.iloc[:,1:].values)
    X_diff = torch.Tensor(df_diff.iloc[:,1:].values)
    y = torch.Tensor(df_labels.iloc[:,1].values)

    return TensorDataset(X_dti, X_rs, X_other, X_cort, X_diff, y)

def create_datasets(dataset, batch_size, num_workers=0, valid_size=0.2, shuffle=True):
    # obtain training indices that will be used for validation
    num_data = len(dataset)
    if shuffle:
        indices = list(range(num_data))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_data))
    valid_idx, train_idx = indices[:split], indices[split:]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # load training data in batches
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers)
    
    # load validation data in batches
    valid_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=num_workers)
    
    return train_loader, valid_loader

def train_epoch_single(model, dataloader, index, criterion, optimizer, scheduler, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    count = 0

    piter = tqdm(dataloader, desc='Train', unit='batch', position=1, leave=False)
    for data in piter:
        
        inputs = data[index]
        targets = data[-1]
        inputs = inputs.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)
        nxt_count = count+batch_size
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # statistics
        epoch_loss = loss.item() * batch_size/nxt_count + epoch_loss * count/nxt_count
        epoch_acc = ((preds == targets).sum()/np.prod(preds.size())).item() * batch_size/nxt_count + epoch_acc * count/nxt_count
        
        count = nxt_count
        piter.set_postfix(accuracy=100. * epoch_acc)

    scheduler.step()
    train_stats = {
        'train_loss': epoch_loss,
        'train_acc': 100. * epoch_acc,
    }
    
    return model, train_stats

def test_single(model, dataloader, index, device):
    since = time.time()
    model.eval()   # Set model to evaluate mode
    
    corrects = 0
    count = 0

    # Iterate over data.
    with torch.no_grad():
        piter = tqdm(dataloader, desc='Test', unit='batch')
        for data in piter:

            inputs = data[index]
            targets = data[-1]
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            batch_size = inputs.size(0)
            count += batch_size

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            corrects += torch.sum(preds == targets.data)/np.prod(preds.size())*batch_size

    acc = corrects.double().item() / count

    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, Test Acc: {100. * acc}')
    
    test_stats = {
        "test_acc": 100. * acc,
    }

    return test_stats
        
if __name__=='__main__':
    DATA_PATH = '../data/'
    dti_file = os.path.join(DATA_PATH, 'DTIConnectData.csv')
    rsfmri_file = os.path.join(DATA_PATH, 'restingstatedata.csv')
    other_file = os.path.join(DATA_PATH, 'otherdata.csv')
    cort_file = os.path.join(DATA_PATH, 'corticalthickness.csv')
    diff_file = os.path.join(DATA_PATH, 'diffusivity.csv')
    outcome_file = os.path.join(DATA_PATH, 'outcome.csv')
    
    abcd_dataset = get_abcd(dti_file, rsfmri_file, other_file, cort_file, diff_file, outcome_file)
    dataloader = DataLoader(abcd_dataset, batch_size=20, shuffle=False)
    dti, rs, other, cort, diff, y = next(iter(dataloader))
    print(dti.shape, rs.shape, other.shape, cort.shape, diff.shape, y.shape)