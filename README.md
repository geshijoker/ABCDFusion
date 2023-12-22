# ABCDFusion
This is the repository to build ML model on multiview data including DTI connectivity data, resting state data, Corticalthichness and other data from ABCD dataset to predict "becoming CHR after 3 years". 
This README.md assume you're using an Unix/macOS system.

# Repo Structure

    .
    ├── abcdfusion              # Source files, define the major modules to build machine learning pipeline with pytorch
    ├── data                    # Data folder, store raw data (.xlsx) and data after preprocessing (.csv)
    ├── data_engineer           # Source files, inspect raw data and conduct preprocessing
    ├── examples                # Source files, perform group-wise (sites) cross validation on single/multiple modalities 
    ├── notebooks               # Source files, notebooks to get preliminary results with sklearn 
    ├── configs.yaml            # Config file, define the model architectures and data loading path
    ├── LICENSE
    └── README.md

# Install Environment
## Install a list of requirements specified in a Requirements File.
```console
foo@bar:~$ python3 -m pip install -r requirements.txt
```
## Force to install all packages even if they are already up-to-date.
```console
foo@bar:~$ pip install -r requirements.txt --force-reinstall 
```
## Install the requirments and ignore the installed packages/files
```console
foo@bar:~$ pip install -r requirements.txt --ignore-installed
```

# Installa from Source Code
## Ensure pip, setuptools, and wheel are up to date
```console
foo@bar:~$ python3 -m pip install --upgrade pip setuptools wheel
```
## Create a virtual environment
```console
foo@bar:~$ python3 -m venv abcd_env
foo@bar:~$ source abcd_env/bin/activate
```
## Install normally from src.
```console
foo@bar:~$ python3 -m pip install .
```
## Install in development mode, i.e. in such a way that the project appears to be installed, but yet is still editable from the src tree.
```console
foo@bar:~$ python3 -m pip install -e .
```

# Use Tensorboard to Visualize the Training Procedures and Recorded Statistics
## Installation
```console
foo@bar:~$ pip install tensorboard
foo@bar:~$ tensorboard --logdir=runs
```
## Start from a local computer if the stats are saved in folder_to_exps
```console
foo@bar:~$ tensorboard --logdir=folder_to_exps
```
## Start from a remote computer if the stats are saved in folder_to_exps
```console
foo-remote@bar-remote:~$ tensorboard --logdir folder_to_exps --port 6006
foo@bar:~$ ssh -N -f -L localhost:16006:localhost:6006 foo-remote@bar-remote
```
