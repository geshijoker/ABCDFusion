import os
import copy
import yaml

import numpy as np
from sklearn import preprocessing

def get_preprocess(arrays, modalities, group_site=True, combine=True):
    # X_dti, X_rs, X_other, X_cort, X_anno
    X_anno = arrays.pop()
    if group_site:
        groups = X_anno[:,-1]
    y, age, sex, site = preprocess_anno(X_anno)
        
    X = []
    if combine:
        if 'DTI' in modalities:
            X.append(preprocess_dti(arrays[0], [age, sex]))
        if 'RS' in modalities:
            X.append(preprocess_rs(arrays[1], [age, sex]))
        if 'OTHER' in modalities:
            X.append(preprocess_other(arrays[2], [sex]))
        if 'CORT' in modalities:
            X.append(preprocess_cort(arrays[3], [age, sex]))
    else:
        if 'DTI' in modalities:
            X.append(preprocess_dti(arrays[0]))
        if 'RS' in modalities:
            X.append(preprocess_rs(arrays[1]))
        if 'OTHER' in modalities:
            X.append(preprocess_other(arrays[2]))
        if 'CORT' in modalities:
            X.append(preprocess_cort(arrays[3]))
    
    if group_site:
        return X, y, groups
    else:
        return X, y

def preprocess_dti(X_dti, extra_features=None, max_lim=1, min_lim=0):
    X_std = (X_dti - 0) / (1 - 0)
    X_scaled = X_std * (max_lim - min_lim) - min_lim
    
    if extra_features:
        X_extra = np.concatenate(extra_features, axis=1)
        X = np.concatenate((X_scaled, X_extra), axis=1)
    else:
        X = X_scaled
    return X

def preprocess_rs(X_rs, extra_features=None, max_lim=1, min_lim=0):
    X_std = (X_rs - (-0.5)) / (2 - (-0.5))
    X_scaled = X_std * (max_lim - min_lim) - min_lim
    
    if extra_features:
        X_extra = np.concatenate(extra_features, axis=1)
        X = np.concatenate((X_scaled, X_extra), axis=1)
    else:
        X = X_scaled
    return X

def preprocess_other(X_other, extra_features=None):
    X_scaled = copy.deepcopy(X_other)
    
    age_scaler = preprocessing.MinMaxScaler()
    X_scaled[:,0:1] = age_scaler.fit_transform(X_other[:,0:1])
    
    X_scaled[:,1:2] = preprocessing.minmax_scale(X_other[:,1:2])
    X_scaled[:,2:3] = preprocessing.minmax_scale(X_other[:,2:3])
    
    X_scaled[:,3:4] = preprocessing.quantile_transform(X_other[:,3:4], n_quantiles=20, copy=True)
    X_scaled[:,4:5] = preprocessing.quantile_transform(X_other[:,4:5], n_quantiles=20, copy=True)
    if extra_features:
        X_extra = np.concatenate(extra_features, axis=1)
        X = np.concatenate((X_scaled, X_extra), axis=1)
    else:
        X = X_scaled
    return X

def preprocess_cort(X_cort, extra_features=None, max_lim=1, min_lim=0):
    X_std = (X_cort - 0) / (5 - 0)
    X_scaled = X_std * (max_lim - min_lim) - min_lim
    
    if extra_features:
        X_extra = np.concatenate(extra_features, axis=1)
        X = np.concatenate((X_scaled, X_extra), axis=1)
    else:
        X = X_scaled
    return X

def preprocess_anno(anno):
    label, age, sex, site = anno[:,0:1], anno[:,1:2], anno[:,2:3], anno[:,3:4]
    
    age_scaler = preprocessing.MinMaxScaler()
    age = age_scaler.fit_transform(age)
    
    sex_scaler = preprocessing.LabelBinarizer()
    sex = sex_scaler.fit_transform(sex)
    
    site_scaler = preprocessing.OneHotEncoder(handle_unknown='ignore')
    site_scaler.fit_transform(site)
    
    return label, age, sex, site
