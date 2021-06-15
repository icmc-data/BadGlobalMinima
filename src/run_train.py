# Flexible integration for any Python script
import wandb
import optax
from train import train
import datasets
from resnet import ResNet18

# 1. Start a W&B run
wandb.init(project='bad-global-minima', entity='data-icmc')

net = ResNet18
wandb.config.net = net

wandb.config.initial_lr = 1e-1
wandb.config.lr_boundaries = [150, 250]

wandb.config.seed = 0
wandb.config.augmentation = False
wandb.config.epochs = 1
wandb.config.batch_size = 128
wandb.config.l2 = True
wandb.config.momentum = True

wandb.config.adversarial_dataset = False
wandb.config.R = None
wandb.config.zero_out_ratio = None



if __name__ == "__main__":
    if wandb.config.adversarial_dataset:
        dataloader = datasets.get_adversarial_cifar(data_root='.', download_data=True, split = 'train', batch_size = wandb.config.batch_size, R = wandb.config.R, zero_out_ratio = wandb.config.zero_out_ratio)
        dataloader_test = None
    else:
        dataloader = datasets.get_cifar(data_root='.', download_data=True, split = 'train', batch_size = wandb.config.batch_size, augmentation = wandb.config.augmentation)
        dataloader_test = datasets.get_cifar(data_root='.', split = 'test', batch_size = wandb.config.batch_size)
    
    boundaries_and_scales = {ep * len(dataloader) : 1/10 for ep in wandb.config.lr_boundaries}
    schedule_fn = optax.piecewise_constant_schedule(-wandb.config.initial_lr, boundaries_and_scales)
    train(net, wandb.config.epochs, dataloader, dataloader_test, schedule_fn, wandb.config.l2, wandb.config.momentum, wandb.config.seed)
