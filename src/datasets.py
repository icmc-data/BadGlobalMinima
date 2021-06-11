import numpy as np
import torch

from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torchvision import transforms


NORMALIZE_TRANSFORM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

AUGMENTATION_TRANSFORMS = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    NORMALIZE_TRANSFORM,
])

STANDARD_TRANFORMS = transforms.Compose([
    transforms.ToTensor(),
    NORMALIZE_TRANSFORM
])

def get_adversarial_cifar(data_root, download_data=False, R=1, zero_out_ratio=0, random_state=0):
    ds = CIFAR10(root=data_root,
                 download=download_data,
                 train=True)

    adv_ds = AdversarialDataset(ds.data, 
                                n_classes=10, 
                                R=R, 
                                zero_out_ratio=zero_out_ratio, 
                                random_state=random_state)
    
    return adv_ds


def get_cifar(data_root, download_data=False, split='train', augmentation=False):
    if split not in ['train', 'test']:
        raise ValueError(f'split must be train or test, not {split}')

    transform = AUGMENTATION_TRANSFORMS if augmentation else STANDARD_TRANFORMS

    ds = CIFAR10(root=data_root, 
                 train=True if split == 'train' else False,
                 download=download_data,
                 transform=transform)

    return ds

class AdversarialDataset(Dataset):
    def __init__(self, original_images, n_classes=10, R=1, zero_out_ratio=0, random_state=0):
        self.n_classes = n_classes
        self.R = R
        self.zero_out_ration = zero_out_ratio
        self.transforms = STANDARD_TRANFORMS

        np.random.seed(random_state)
        
        self.n = original_images.shape[0]*R
        self.y = np.random.randint(n_classes, size=self.n)
        self.x = np.repeat(original_images, R, axis=0)
        
        # Add noise to x
        # The authors perform per RGB value zeroing-out; NOT per pixel
        # https://github.com/chao1224/BadGlobalMinima/blob/950b0c0e2dc0ab2bda5077e81e6ad317e87c5a18/cifar10/cifar10_data/confusion_data_generation.py#L93
        if zero_out_ratio > 0:
            flatten_x = self.x.reshape(self.x.shape[0], -1)
            n_pos = flatten_x.shape[1]
            n_per_image_zeros = int(zero_out_ratio*n_pos)

            img_idx = np.repeat(np.arange(self.n), n_per_image_zeros)
            pos_idx = np.random.randint(n_pos, size=(self.n*n_per_image_zeros))

            flatten_x[img_idx, pos_idx] = 0
            self.x = flatten_x.reshape(self.x.shape)
            
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        x = self.transforms(x)

        return x, y
    
    def __len__(self):
        return self.n