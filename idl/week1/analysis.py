from collections.abc import Callable, Iterable

import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_learning_curves(metrics: dict[str, np.ndarray],
                         keys: Iterable[str]):
    """Basic plots for metrics of interest.

    Parameters:
        metrics: Dictionary as returned by a Trainer object's train_model function.
        keys: Plots are made for each metric named in here. Each plot gets one line for training and one for validation.
    """
    for key in keys:
        plt.figure(figsize=(12, 3))
        plt.plot(metrics["train_" + key], label="train")
        plt.plot(metrics["val_" + key], label="validation")
        plt.legend()
        plt.title(key)
        plt.xlabel("Epoch")
        plt.show()


def confusion_matrix(model: Callable[[torch.Tensor], torch.Tensor],
                     inputs: np.ndarray,
                     labels: np.ndarray,
                     num_classes: int | None = None,
                     device: str = "cpu") -> np.ndarray:
    """Basic confusion matrix for classifiers.

    Rows contain correct classes. Columns contain predicted classes. Each cell gives: For inputs that have the actual
    class given in this row, how often was the class in this column predicted?
    
    Parameters:
        model: The classifier model, taking in tensors and returning per-class scores.
        inputs: Array containing inputs to compute the confusion matrix for.
        labels: Corresponding true labels for inputs.
        num_classes: How many classes there are. If not given, this is inferred by the largest label given.
        device: Which device the inputs should be pushed to. Must match model device.
    """
    if num_classes is None:
        num_classes = labels.max().item() + 1
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    with torch.no_grad():
        train_predictions = torch.argmax(model(torch.tensor(inputs, device=device)), axis=-1).numpy(force=True)
    for class_index in range(num_classes):
        predictions_for_this_class = train_predictions[labels == class_index]
        for class_index_column in range(num_classes):
            count = len(np.where(predictions_for_this_class == class_index_column)[0])
            matrix[class_index, class_index_column] = count
    return matrix


def precision_recall(confusion_matrix: np.ndarray):
    """Get per-class precision/recall from confusion matrix.
    
    For each class, precision is the proportion of correct responses out of all the times this class was predicted.
    Recall is the proportion of correct responses out of all times this class was actually the true one.
    """
    n_classes = confusion_matrix.shape[0]
    for class_index in range(n_classes):
        recall = confusion_matrix[class_index, class_index] / confusion_matrix[class_index].sum()
        precision = confusion_matrix[class_index, class_index] / confusion_matrix[:, class_index].sum()
        print("Class", class_index)
        print(f"Recall: {recall:.5f}")
        print(f"Precision: {precision:.5f}")
