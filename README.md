# ABCDFusion
This is the repo to build ML model on multiview data including DTI connectivity data, resting state data and other data from ABCD dataset prediction "becoming CHR after 3 years". 

# Install Environment
## Force to install all packages even if they are already up-to-date.
```console
foo@bar:~$ pip install -r requirements.txt --force-reinstall 
```
## Install the requirments and ignore the installed packages/files
```console
foo@bar:~$ pip install -r requirements.txt --ignore-installed
```
## Use tensorboard to visualize the training procedures and stats
### Installation
```console
foo@bar:~$ pip install tensorboard
tensorboard --logdir=runs
```
### Start from a local computer if the stats are saved in exps
```console
foo@bar:~$ tensorboard --logdir=exps
```
### Start from a remote computer if the stats are saved in exps
```console
foo-remote@bar-remote:~$ tensorboard --logdir exps --port 6006
foo@bar:~$ ssh -N -f -L localhost:16006:localhost:6006 foo-remote@bar-remote
```