from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.v2 import Compose, ToDtype, ToImage, Transform

from .utils import plot_image_grid


def get_datasets_and_loaders(dataset: str,
                             batch_size: int,
                             num_workers: int = 0,
                             root: str = "data",
                             augment_transforms: list[Transform] | None = None,
                             verbose: bool = True) \
                                -> tuple[datasets.VisionDataset,
                                         datasets.VisionDataset,
                                         DataLoader,
                                         DataLoader]:
    """Standard preparation of datasets (train/validation) and data loaders.

    Only for vision datasets.

    Parameters:
        dataset: Name of the dataset. Currently allowed are mnist, fashion, cifar10, cifar100.
                 mnist: You know; MNIST. Handwritten digits 0-9. Gets padded to 32x32.
                 fashion: FashionMNIST, structure like MNIST, but data is fashion items like shirts, shoes, bags...
                 cifar10: Should know this one; 32x32 color images of various animals and vehicles.
                 cifar100: CIFAR but with 100 classes.
        batch_size: Guess what!
        num_workers: Used by DataLoader.
        root: Base path where datasets should be stored/looked for.
        augment_transforms: Transforms you want to be applied only to the training data, i.e. data augmentation.
        verbose: If True, print some info about the dataset elements (shape and dtype), as well as plotting some example
                 images.
    """
    # torch keeps telling me to use this instead of ToTensor...
    to_image = ToImage()
    train_transforms = [to_image]
    test_transforms = [to_image]
    to_float = ToDtype(torch.float32, scale=True)
    train_transforms.append(to_float)
    test_transforms.append(to_float)

    if augment_transforms is not None:
        train_transforms += augment_transforms

    train_transforms = Compose(train_transforms)
    test_transforms = Compose(test_transforms)

    if dataset == "mnist":
        constructor = datasets.MNIST
    elif dataset == "fashion":
        constructor = datasets.FashionMNIST
    elif dataset == "cifar10":
        constructor = datasets.CIFAR10
    elif dataset == "cifar100":
        constructor = datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset {dataset}!")
    train_data = constructor(root=root, train=True, transform=train_transforms, download=True)
    test_data = constructor(root=root, train=False, transform=test_transforms, download=True)
        
    # NOTE drop_last=True for training drops the last "leftover" batch if it's smaller than the batch size
    # -> guarantees consistent batch size always
    train_dataloader = DataLoader(train_data, batch_size=batch_size, pin_memory=True,
                                  num_workers=num_workers, persistent_workers=num_workers > 0,
                                  drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True,
                                 num_workers=num_workers)
    if verbose:
        images, y = next(iter(train_dataloader))
        print(f"Shape/dtype of input batch X [N, C, H, W]: {images.shape}, {images.dtype}")
        print(f"Min/max pixel values in input batch: {images.min().item():.3g}, {images.max().item():.3g}")
        print(f"Shape/dtype of label batch y: {y.shape}, {y.dtype}")
        plot_image_grid(images[:128],
                        figure_size=(14, 7), title="Example images", n_rows=8, n_cols=16)
    return train_data, test_data, train_dataloader, test_dataloader
