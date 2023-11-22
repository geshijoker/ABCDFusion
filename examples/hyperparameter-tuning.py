import time
import datetime
import random
import sys
import os
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from abcdfusion import metrics
from abcdfusion.models import MLPNet
from abcdfusion.data_utils import get_datset, get_default_transforms
from abcdfusion.utils import check_make_dir
from abcdfusion import train_epoch, test

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--data', '-d', type=str, required=True,
                    help='data folder to load data')
parser.add_argument('--seed', '-s', type=int, default=None, 
                    help='which seed for random number generator to use')
parser.add_argument('--architecture', '-a', type=str, default='unet',
                    help='model architecture')
parser.add_argument('--loss', '-l', type=str, default='ce',
                    help='the loss function to use')
parser.add_argument('--test_while_train', '-t', action='store_true',
                    help='using test while train')
parser.add_argument('--benchmark', action='store_true',
                    help='using benchmark algorithms')
parser.add_argument('--debug', action='store_true',
                    help='using debug mode')
parser.add_argument('--verbose', action='store_true',
                    help='verbose mode')

# load and parse argument
args = parser.parse_args()

if args.gpu<0 or not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    if args.gpu<torch.cuda.device_count():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device("cuda") 
print('Using device: {}'.format(device))

# set up the seed
if args.seed:
    seed = args.seed
else:
    seed = torch.seed()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

x = torch.rand(batch_size, 3, input_height, input_width)

if args.architecture == 'unet':
    model = UNet(downward_params, upward_params, output_params)
elif args.architecture == 'unet-crf':
    unet = UNet(downward_params, upward_params, output_params)
    model = nn.Sequential(
        unet,
        CRF(n_spatial_dims=2)
    )
out = model(x)
print('output shape', out.shape) 
    
net = Net(config["l1"], config["l2"])

checkpoint = session.get_checkpoint()

if checkpoint:
    checkpoint_state = checkpoint.to_dict()
    start_epoch = checkpoint_state["epoch"]
    net.load_state_dict(checkpoint_state["net_state_dict"])
    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
else:
    start_epoch = 0
    
optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
net.to(device)

def train_tune(config, data_dir=None):
    print('Starting training loop; initial compile can take a while...')
    since = time.time()
    model.train()   # Set model to evaluate mode
    start_epoch = 0

    pbar = trange(num_epochs, desc='Epoch', unit='epoch', initial=start_epoch, position=0)
    # Iterate over data.
    for epoch in pbar:
        model, epoch_loss, epoch_acc, train_stats = train_epoch(model, train_loader, n_classes, criterion, optimizer, scheduler, device)
        if test_while_train:
            cl_wise_iou, test_stats = test(model, test_loader, n_classes, device)

        if writer:
            writer.add_scalar('time eplased', time.time() - since, epoch)
            for stat in train_stats:
                writer.add_scalar(stat, train_stats[stat], epoch)
            if test_while_train:
                for stat in test_stats:
                    writer.add_scalar(stat, test_stats[stat], epoch)
                for cl_i in range(len(cl_wise_iou)):
                    writer.add_scalar(f'class_{classes[cl_i]}_iou', cl_wise_iou[cl_i], epoch)

        pbar.set_postfix(loss = epoch_loss, acc = epoch_acc)

        if epoch+1==num_epochs or (frequency>0 and epoch%frequency==0):
            save_checkpoint(epoch)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, last epoch loss: {epoch_loss}, acc: {epoch_acc}')


session.report(
    {"loss": val_loss / val_steps, "accuracy": correct / total},
)

config = {
    "l1": tune.choice([2 ** i for i in range(9)]),
    "l2": tune.choice([2 ** i for i in range(9)]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16])
}

gpus_per_trial = 2
# ...
result = tune.run(
    partial(train_tune, data_dir=data_dir),
    resources_per_trial={"cpu": 24, "gpu": 1},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    checkpoint_at_end=False)

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 24, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)