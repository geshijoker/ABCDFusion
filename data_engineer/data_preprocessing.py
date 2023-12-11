import os
import errno
import sys
import time

import pandas as pd

ROOT_PATH = '../'
DATA_PATH = 'data/'
ORIGINAL_DATA = 'RawData.xlsx'

# load raw data
orig_data_path = os.path.join(ROOT_PATH, DATA_PATH, ORIGINAL_DATA)

if os.path.isfile(orig_data_path):
    dfs = pd.read_excel(orig_data_path, sheet_name=None)
else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), orig_data_path)
    
sheet_names = dfs.keys()
print('Raw data loaded. The loaded panda frames are', sheet_names)

# DTIconnectivity data
dfs['DTIconnectivity'] = dfs['DTIconnectivity'].dropna()
dfs['DTIconnectivity'] = dfs['DTIconnectivity'][dfs['DTIconnectivity']['imgincl_dmri_include'] == 1] 
dfs['DTIconnectivity'] = dfs['DTIconnectivity'].loc[dfs['DTIconnectivity']['age'] >= 10]
dfs['DTIconnectivity'].drop('imgincl_dmri_include', inplace=True, axis=1)
dfs['DTIconnectivity'].rename(columns={'sex_1isM': 'sex'}, inplace=True)

# RestingState data
dfs['RestingState'] = dfs['RestingState'].dropna()
dfs['RestingState'] = dfs['RestingState'][dfs['RestingState']['imgincl_rsfmri_include'] == 1]
dfs['RestingState'] = dfs['RestingState'].loc[dfs['RestingState']['age'] >= 10]
dfs['RestingState'].drop('imgincl_rsfmri_include', inplace=True, axis=1)
dfs['RestingState'].rename(columns={'sex_1isM': 'sex'}, inplace=True)

# Corticalthickness data
dfs['Corticalthickness'] = dfs['Corticalthickness'].dropna()
dfs['Corticalthickness'] = dfs['Corticalthickness'][dfs['Corticalthickness']['imgincl_t1w_include'] == 1] 
dfs['Corticalthickness'] = dfs['Corticalthickness'].loc[dfs['Corticalthickness']['yr2_age'] >= 10]
dfs['Corticalthickness'].drop(['eventname','imgincl_t1w_include'], inplace=True, axis=1)
dfs['Corticalthickness'].rename(columns={'yr2_age': 'age'}, inplace=True)

# other data
dfs['otherdata'] = dfs['otherdata'].dropna()
dfs['otherdata'] = dfs['otherdata'].loc[dfs['otherdata']['ageat2yr'] >= 10]
dfs['otherdata'].rename(columns={'ageat2yr': 'age'}, inplace=True)

# other data
dfs['outcome'] = dfs['outcome'].dropna()
dfs['outcome'].drop(dfs['outcome'].iloc[:, 2:], axis = 1, inplace=True)

# select shared data
shared_subjects = set(dfs['outcome']['src_subject_id']).intersection(\
    set(dfs['DTIconnectivity']['src_subject_id']),\
    set(dfs['RestingState']['src_subject_id']),\
    set(dfs['otherdata']['src_subject_id']),\
    set(dfs['Corticalthickness']['src_subject_id']))

filtered_dfs = {sheet_name: df[df['src_subject_id'].isin(shared_subjects)] for sheet_name, df in dfs.items()}

for sheet_name, df in filtered_dfs.items():
    df.sort_values(by=['src_subject_id'], inplace=True)
    df.set_index('src_subject_id', inplace=True)
    print(sheet_name, '{} of subjects are sorted, with {} nan'.format(len(df)-1, df.isnull().values.any()))
    
assert filtered_dfs['DTIconnectivity']['age'].equals(filtered_dfs['RestingState']['age']) and filtered_dfs['RestingState']['age'].equals(filtered_dfs['Corticalthickness']['age']) and filtered_dfs['Corticalthickness']['age'].equals(filtered_dfs['otherdata']['age']), 'ages do not match'

assert filtered_dfs['DTIconnectivity']['sex'].equals(filtered_dfs['RestingState']['sex']) and filtered_dfs['RestingState']['sex'].equals(filtered_dfs['Corticalthickness']['sex']), 'sex do not match'

assert filtered_dfs['DTIconnectivity']['site'].equals(filtered_dfs['RestingState']['site']) and filtered_dfs['RestingState']['site'].equals(filtered_dfs['Corticalthickness']['site']), 'sites do not match'

# Merge age, sex, site
filtered_dfs['outcome']['age'] = filtered_dfs['DTIconnectivity']['age']
filtered_dfs['outcome']['sex'] = filtered_dfs['DTIconnectivity']['sex']
filtered_dfs['outcome']['site'] = filtered_dfs['DTIconnectivity']['site']

filtered_dfs['DTIconnectivity'].drop(['age', 'sex', 'site'], axis=1, inplace=True)
filtered_dfs['RestingState'].drop(['age', 'sex', 'site'], axis=1, inplace=True)
filtered_dfs['Corticalthickness'].drop(['age', 'sex', 'site'], axis=1, inplace=True)

# output to file
for sheet_name, df in filtered_dfs.items():
    print('check the column of {}'.format(sheet_name), df.columns.values[:4])
    output_file = '{}.csv'.format(sheet_name)
    df.to_csv(os.path.join(ROOT_PATH, DATA_PATH, output_file), index=False)
    print('{} is saved'.format(sheet_name))