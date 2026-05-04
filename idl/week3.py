
import torch
from matplotlib import pyplot as plt


def visualize_features(features: torch.Tensor,
                       n_rows: int,
                       data_shape: tuple[int, ...],
                       colormap: str = "local",
                       normalization: str = "symmetric",
                       figure_size: tuple[int, int] = (12, 12)):
    """Used to visualize first-layer features of an MLP.
    
    Parameters:
        features: Weights to visualize.
        n_rows: We visualize a grid of n_rows x n_rows many features. If there are more features in the input, they are
                ignored. If there are fewer, there will be empty spots.
        data_shape: Original shape of the data. For example, this may be (3, 32, 32) for CIFAR images.
        colormap: One of 'global' or 'local'. 'global' uses one colormap for all features. 'local' uses one colormap per
                  feature.
        normalization: One of of 'symmetric' or 'full'. 'symmetric' will map values such that a weight of 0 becomes 0.5,
                       giving it "medium brightness". If weight values are sweked, this will result in the colormap not
                       being fully utilized. 'full' maps values that they use the use the full colormap range.
        figure_size: Size of the overall plot.
    """
    if colormap not in ["local", "global"]:
        raise ValueError("colormap argument should be 'local' or 'global'")
    if normalization not in ["symmetric", "full"]:
        raise ValueError ("normalization should be 'symmetric' (map 0 weights to 0.5) "
                          "or 'full' (map minimum to 0 and maximum to 1)")
    
    features = features[:n_rows**2]
    if colormap == "global":
        if normalization == "full":
            features -= features.min()
            features /= features.max()
        else:
            absmax = abs(features).max()
            features /= 2*absmax
            features += 0.5

    plt.figure(figsize=figure_size)
    for ind, pattern in enumerate(features):
        if colormap == "local":
            if normalization == "full":
                pattern -= pattern.min()
                pattern /= pattern.max()
            else:
                absmax = abs(pattern).max()
                pattern /= 2*absmax
                pattern += 0.5
        
        plt.subplot(n_rows, n_rows, ind+1)
        pattern = pattern.reshape(*data_shape).transpose((1, 2, 0))
        plt.imshow(pattern)
        plt.axis("off")
        #plt.colorbar()
    plt.suptitle("First layer features with {} colormaps and {} normalization".format(colormap, normalization))
    plt.show()
