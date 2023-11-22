import csv
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, TensorDataset, DataLoader

from tqdm import tqdm, trange
from abcdfusion import metrics

def get_abcd(DTI_csv, rsfMRI_csv, other_csv, label_csv):
    df_dti = pd.read_csv(DTI_csv)
    df_rs = pd.read_csv(rsfMRI_csv)
    df_other = pd.read_csv(other_csv)
    df_labels = pd.read_csv(label_csv)

    X_dti = torch.Tensor(df_dti.iloc[:,3:].values)
    X_rs = torch.Tensor(df_rs.iloc[:,2:].values)
    X_other = torch.Tensor(df_other.iloc[:,1:].values)
    y = torch.Tensor(df_labels.iloc[:,1:].values)

    return TensorDataset(X_dti, X_rs, X_other, y)

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    count = 0

    piter = tqdm(dataloader, desc='Batch', unit='batch', position=1, leave=False)
    for inputs, targets in piter:

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

    epoch_acc *= 100.
    scheduler.step()
    train_stats = {
        'train_loss': epoch_loss,
        'train_acc': epoch_acc,
    }
    
    return model, epoch_loss, epoch_acc, train_stats

def test(model, dataloader, device):
    since = time.time()
    model.eval()   # Set model to evaluate mode
    
    corrects = 0
    count = 0

    # Iterate over data.
    with torch.no_grad():
        piter = tqdm(dataloader, unit='batch')
        for inputs, targets in piter:
            piter.set_description(f"Test ")

            inputs = inputs.to(device)
            targets = targets.to(device)
            
            batch_size = inputs.size(0)
            count += batch_size

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            corrects += torch.sum(preds == targets.data)/np.prod(preds.size())*batch_size

    acc = corrects.double().item() / count
    homo = homogeneity.double().item() / count

    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, Test Acc: {100. * acc}, Test Iou: {mean_IOU}')
    
    test_stats = {
        "test_acc": 100. * acc,
    }

    return cl_wise_iou, test_stats

if __name__ == '__main__':
    pass

        
if __name__=='__main__':
    ROOT_PATH = '../data/'
    dti_file = os.path.join(ROOT_PATH, 'DTIConnectData.csv')
    rsfmri_file = os.path.join(ROOT_PATH, 'restingstatedata.csv')
    other_file = os.path.join(ROOT_PATH, 'otherdata.csv')
    outcome_file = os.path.join(ROOT_PATH, 'outcome.csv')
    
    abcd_dataset = get_abcd(dti_file, rsfmri_file, other_file, outcome_file)
    dataloader = DataLoader(abcd_dataset, batch_size=20, shuffle=False)
    dti, rs, other, y = next(iter(dataloader))
    print(dti.shape, rs.shape, other.shape, y.shape)
    
    
    