# Flexible integration for any Python script
import wandb
from train import train
import datasets
from resnet import ResNet18


wandb.config.epochs = 10


# 1. Start a W&B run
wandb.init(project='BadLocalMinima', entity='joaopedromattos')


if __name__ == "__main__":
    dataloader = datasets.get_cifar(data_root='.', download_data=True, split = 'train')
    dataloader_test = datasets.get_cifar(data_root='.', split = 'test')

    net = ResNet18
    train("ResNet18_test", "Testing the first version of our code", net, 1, dataloader, dataloader_test)
