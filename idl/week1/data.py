import gzip
import pickle
import requests
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


MNIST_FOLDER = "mnist"
URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"


def get_mnist(root: str = "data"):
    """Download MNIST data if it's not already there.
    
    Adopted from https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
    """
    root = Path(root)
    path = root / MNIST_FOLDER
    path.mkdir(parents=True, exist_ok=True)

    if not (path / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (path / FILENAME).open("wb").write(content)


def load_mnist(root: str = "data"):
    """Load MNIST files from disk.
    
    Adopted from https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
    """
    path = Path(root) / MNIST_FOLDER
    with gzip.open((path / FILENAME).as_posix(), "rb") as file:
        return pickle.load(file, encoding="latin-1")


def mnist_overview(inputs_train: np.ndarray,
                   labels_train: np.ndarray,
                   inputs_valid: np.ndarray,
                   labels_valid: np.ndarray,
                   inputs_test: np.ndarray,
                   labels_test: np.ndarray):
    """Display some statistics for the MNIST dataset.
    
    Would technically also work for other datasets with the same structure given as numpy arrays.
    """
    plt.figure(figsize=(6, 6))
    for ind, img in enumerate(inputs_train[:64]):
        plt.subplot(8, 8, ind+1)
        plt.imshow(img.reshape((28, 28)), cmap="Greys")
        plt.axis("off")
    plt.suptitle("First 64 images from the training set")
    plt.show()

    print(f"Training input shape {inputs_train.shape}; Validation input shape {inputs_valid.shape}; "
          f"Test input shape {inputs_test.shape}")
    print(f"Training labels shape {labels_train.shape}; Validation labels shape {labels_valid.shape}; "
          f"Test labels shape {labels_test.shape}")
    print(f"Data Python type: Inputs {type(inputs_train)}; Labels {type(labels_train)}")
    print(f"Data dtype: Inputs {inputs_train.dtype}; Labels {labels_train.dtype}")

    # What is this strange binning?
    # MNIST is originally stored as 8bit integers in the range [0, 255].
    # the data here has been preprocessed to floats in the range [0, 1],
    # but there are still only 256 unique values. 
    # So we create one bin per value.
    # to actually center around a value, like 0, we need to create a bin like [-0.5, 0.5].
    # so we do this until the top end, [254.5, 255.5] to get a bin for the value 255.
    # then we divide by 255 to normalize to range [0, 1].
    bins = np.arange(-0.5, 256.5, 1) / 255
    plt.figure(figsize=(10, 4))
    plt.hist(inputs_train.reshape(-1), bins=bins)
    plt.title("Pixel distribution (linear)")
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(inputs_train.reshape(-1), bins=bins)
    plt.yscale("log")
    plt.title("Pixel distribution (logarithmic)")
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.show()

    # now the labels
    bins = np.arange(-0.5, 10.5, 1)
    plt.hist(labels_train, bins=bins)
    plt.hlines(5000, -1, 10, colors="red", linestyles="dashed", label="Ideal balance")

    plt.xticks(np.arange(10))
    plt.xlim(-1, 10)
    plt.title("Label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.legend()
    plt.show()
