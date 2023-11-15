import os
import errno
import sys
import time

import pandas as pd

ROOT_PATH = '../'
DATA_PATH = 'data/'
ORIGINAL_DATA = 'DataforGe_v2.xlsx'
DIFnCOR_DATA = 'diffusivityandcorticalthicknessdataforGe.xlsx'

# load raw data
orig_data_path = os.path.join(ROOT_PATH, DATA_PATH, ORIGINAL_DATA)
difncor_data_path = os.path.join(ROOT_PATH, DATA_PATH, DIFnCOR_DATA)

if os.path.isfile(orig_data_path):
    dfs = pd.read_excel(orig_data_path, sheet_name=None)
else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), orig_data_path)
    
sheet_names = dfs.keys()
print('Raw data loaded. The loaded panda frames are', sheet_names)

if os.path.isfile(difncor_data_path):
    difncor_dfs = pd.read_excel(difncor_data_path, sheet_name=None)
else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), difncor_data_path)
    
sheet_names = difncor_dfs.keys()
print('The loaded panda frames are', sheet_names)
difncor_dfs['diffusivity'] = difncor_dfs['diffusivity'].dropna()

# select shared data
shared_subjects = set(dfs['outcome']['src_subject_id']).intersection(\
    set(dfs['DTIConnectData']['src_subject_id']),\
    set(dfs['restingstatedata']['src_subject_id']),\
    set(dfs['otherdata']['src_subject_id']),\
    set(difncor_dfs['diffusivity']['src_subject_id']),\
    set(difncor_dfs['corticalthickness']['src_subject_id']))

filtered_dfs = {sheet_name: df[df['src_subject_id'].isin(shared_subjects)] for sheet_name, df in dfs.items()}

for sheet_name, df in filtered_dfs.items():
    df.sort_values(by=['src_subject_id'])
    print(sheet_name, '{} of subjects are sorted'.format(len(df)-1))
    
filtered_difncor_dfs = {sheet_name: df[df['src_subject_id'].isin(shared_subjects)] for sheet_name, df in difncor_dfs.items()}

print('In the filtered data frames, the number of subjects in each sheet is')
for sheet_name, df in filtered_dfs.items():
    print(sheet_name, len(df))
    
# output to file
for sheet_name, df in filtered_dfs.items():
    output_file = '{}.csv'.format(sheet_name)
    df.to_csv(os.path.join(ROOT_PATH, DATA_PATH, output_file), index=False)
    print('{} is saved'.format(sheet_name))

for sheet_name, df in filtered_difncor_dfs.items():
    output_file = '{}.csv'.format(sheet_name)
    df.to_csv(os.path.join(ROOT_PATH, DATA_PATH, output_file), index=False)
    print('{} is saved'.format(sheet_name))