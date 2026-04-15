import torch


def accuracy(outputs: torch.Tensor,
             labels: torch.Tensor) -> torch.Tensor:
    predictions = torch.argmax(outputs, axis=-1)
    matches = labels == predictions
    return matches.float().mean()
