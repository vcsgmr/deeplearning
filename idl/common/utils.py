from collections.abc import Iterable

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


def plot_image_grid(images: torch.Tensor,
                    figure_size: tuple[int, int],
                    title: str,
                    n_rows: int,
                    n_cols: int | None = None,
                    subtitles: Iterable[str] | None = None,
                    tight_layout: bool = False,
                    colormap="Greys"):
    """Display a rectangular grid of images.

    Parameters:
        images: Batch of images we want to plot. We assume that this has rows*cols many entries. If this is not true,
                this might crash. Sorry not sorry!
        figure_size: The size of the figure.
        title: Will be used as figure title as well as for naming Tensorboard summaries.
        n_rows: Will plot this many rows, and n_rows**2 many examples in total if n_cols is not igven.
        n_cols: Will plot this many columns of examples. Defaults to n_rows.
        subtitles: If given, should be an iterable of strings of the same length as images. Each string will be used as
                   title for the respective image's subplot.
        tight_layout: If True, use plt.tight_layout().
        colormap: Which colormap to use to display images. Only used for single-channel images.
    """
    if n_cols is None:
            n_cols = n_rows
    images = np.clip(images.cpu().numpy(), 0, 1)

    plt.figure(figsize=figure_size)
    for ind, img in enumerate(images):
        plt.subplot(n_rows, n_cols, ind + 1)
        plt.imshow(img.transpose(1, 2, 0), vmin=0, vmax=1, cmap=colormap)
        plt.axis("off")
        if subtitles is not None:
             plt.title(subtitles[ind], fontsize=8)
    plt.suptitle(title)
    if tight_layout:
         plt.tight_layout()
    plt.show()


def accuracy(outputs: torch.Tensor,
             labels: torch.Tensor) -> torch.Tensor:
    predictions = torch.argmax(outputs, axis=-1)
    matches = labels == predictions
    return matches.float().mean()


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
