import csv
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, TensorDataset, DataLoader

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
    
    
    