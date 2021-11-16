import os
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def dataloader(x, y, batch_size):
    if batch_size > len(x):
        raise ValueError('Batch Size too big.')
    num_eg = len(x)
    assert num_eg == len(y)
    for i in range(0, num_eg, batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]

def load_cifar10(train_batch_size, test_batch_size, **_):
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    train_transform = transforms.Compose(normalize)
    test_transform = transforms.Compose(normalize)

    data_dir = os.path.join(CURRENT_DIR, ".data", "cifar10")

    train_dataset = CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=None,
    )

    test_dataset = CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader, len(train_dataset)
