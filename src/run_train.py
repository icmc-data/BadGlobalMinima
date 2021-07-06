import fire
import wandb
import optax

from train import train
import datasets
from resnet import ResNet18


def download_weights_file(run_weights_name):
    weights_file = wandb.restore("run/weights_final.pkl", 
                     run_path=f"data-icmc/bad-global-minima/{run_weights_name}")
    return weights_file.name


def run_experiment(
        initial_lr=1e-1, lr_boundaries=[150, 250], seed=0, augmentation=False,
        epochs=1, batch_size=128, net = ResNet18, l2=True,
        momentum=True, adversarial_dataset=False, R=1, zero_out_ratio=0,
            testing=False, run_weights_name=None,
    ):
    """Runs the experiment with given parameters and auto logs to wandb.
    In case of errors, make sure you've runned 'wandb login'
    
    :param testing - Dummy testing variable to quickly filter out runs in wandb
    :param run_weights_name - Name of the experiment RUN PATH in Wandb to load weights from.
        If None, default weight initialization is used.
        Note that the experiment name is different than it's run path
    """
    # locals is a htly hacky way of getting function arguments - see https://stackoverflow.com/a/582097
    # Must be used before setting any local variables
    wandb.init(project='bad-global-minima', entity='data-icmc', config=locals())

    if run_weights_name is not None:
        weights_file = download_weights_file(wandb.config.run_weights_name)

    if wandb.config.adversarial_dataset:
        dataloader = datasets.get_adversarial_cifar(
            data_root='.', download_data=True, 
            batch_size = wandb.config.batch_size, R = wandb.config.R, 
            zero_out_ratio = wandb.config.zero_out_ratio
        )
        dataloader_test = None
    else:
        dataloader = datasets.get_cifar(
            data_root='.', download_data=True, split = 'train', 
            batch_size = wandb.config.batch_size, augmentation = wandb.config.augmentation
        )
        dataloader_test = datasets.get_cifar(data_root='.', split = 'test', 
            batch_size = wandb.config.batch_size)
    
    boundaries_and_scales = {ep * len(dataloader) : 1/10 for ep in wandb.config.lr_boundaries}
    schedule_fn = optax.piecewise_constant_schedule(-wandb.config.initial_lr, boundaries_and_scales)
    train(net, wandb.config.epochs, dataloader, dataloader_test, schedule_fn, 
          wandb.config.l2, wandb.config.momentum, wandb.config.seed, weights_file,
          wandb.config.run_weights_name)


if __name__ == "__main__":
    fire.Fire(run_experiment)
