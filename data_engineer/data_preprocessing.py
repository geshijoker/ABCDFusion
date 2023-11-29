import os
import errno
import sys
import time

import pandas as pd

ROOT_PATH = '../'
DATA_PATH = 'data/'
ORIGINAL_DATA = 'DataforGe_v2.xlsx'

# load raw data
orig_data_path = os.path.join(ROOT_PATH, DATA_PATH, ORIGINAL_DATA)

if os.path.isfile(orig_data_path):
    dfs = pd.read_excel(orig_data_path, sheet_name=None)
else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), orig_data_path)
    
sheet_names = dfs.keys()
print('Raw data loaded. The loaded panda frames are', sheet_names)

select shared data
shared_subjects = set(dfs['outcome']['src_subject_id']).intersection(\
    set(dfs['DTIConnectData']['src_subject_id']),\
    set(dfs['restingstatedata']['src_subject_id']),\
    set(dfs['otherdata']['src_subject_id']),\
    set(dfs['diffusivity']['src_subject_id']),\
    set(dfs['corticalthickness']['src_subject_id']),\
    set(dfs['corticalthicknesschange_sexsite']['src_subject_id']),\
    set(dfs['DTIConnect_SexSite']['src_subject_id']),\
    set(dfs['restingstatefmri_sexsite']['src_subject_id']))

filtered_dfs = {sheet_name: df[df['src_subject_id'].isin(shared_subjects)] for sheet_name, df in dfs.items()}

for sheet_name, df in filtered_dfs.items():
    df.sort_values(by=['src_subject_id'])
    print(sheet_name, '{} of subjects are sorted'.format(len(df)-1))
    
# output to file
for sheet_name, df in filtered_dfs.items():
    output_file = '{}.csv'.format(sheet_name)
    df.to_csv(os.path.join(ROOT_PATH, DATA_PATH, output_file), index=False)
    print('{} is saved'.format(sheet_name))