from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .utils import accuracy, plot_image_grid


class TrainerBase:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_loader: DataLoader, 
                 validation_loader: DataLoader,
                 n_epochs: int,
                 device: str,
                 logger: TensorboardLogger | None = None,
                 verbose: bool = True,
                 use_tqdm: bool = False):
        """Base class for training generative models.

        Any Trainer for a specific kind of model should inherit fromt his and implement the core_step function.

        Parameters:
            model: The model to train.
            optimizer: torch optimizer instance.
            training_loader, validation_loader: Dataloaders for training/validation sets.
            n_epochs: Number of full iterations over the training loader.
            device: Device on which all the torch stuff should happen (e.g. "cuda").
            logger: Optional TensorboardLogger instance for logging with Tensorboard.
            verbose: If True, report on training progress throughout.
            use_tqdm: If True, and verbose is also True, supply per-epoch progress bars.
        """
        self.model = model
        self.optimizer = optimizer

        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.n_epochs = n_epochs
        self.device = device

        self.verbose = verbose
        self.use_tqdm = use_tqdm

        self.logger = logger
        self.global_step = 0

        self.full_metrics = defaultdict(list)

    def train_model(self) -> dict[str, np.ndarray]:
        """The main training & evaluation loop + housekeeping.

        Returns:
            Dictionary with training and evaluation metrics per epoch. This maps each train/val metric name to a numpy
            array of per-epoch results. NOTE, for training metrics we only track the average over the epoch.
            This is somewhat imprecise, as the model changes over the epoch. So the metrics at the end of the epoch will
            usually be better than at the start, but we average over everything. A "clean" approach would evaluate on
            the training set at the end of each epoch, but this would take significantly longer.
        """
        n_total_steps = self.n_epochs * len(self.training_loader)
        print(f"Running {n_total_steps} training steps -- {self.n_epochs} epochs at {len(self.training_loader)} steps "
              "per epoch.")
        
        self.optimizer.zero_grad()  # safety first!
        for epoch_ind in tqdm(iterable=range(self.n_epochs), desc="Overall progress", leave=True,
                              disable=not self.use_tqdm or not self.verbose):
            epoch_train_metrics = self.train_epoch(epoch_ind)
            self.finish_epoch(epoch_train_metrics, epoch_ind)

        self.model.eval()
        # TODO make this prettier
        for key in self.full_metrics:
            self.full_metrics[key] = np.array(self.full_metrics[key])
        return dict(self.full_metrics)
    
    def train_epoch(self,
                    epoch_ind: int) -> dict[str, list[torch.Tensor]]:
        """One epoch training loop. Iterates over the training dataloader once and collects metrics.

        Returns:
            Dictionary mapping metric names to lists of per-batch results.
        """
        if self.verbose:
            print(f"Starting epoch {epoch_ind + 1}...", end=" ")
        start_time = perf_counter()
        epoch_train_metrics = defaultdict(list)
        
        self.model.train()
        # manual progressbar required due to multiprocessing in dataloaders
        with tqdm(total=len(self.training_loader), desc=f"Training", leave=False,
                  disable=not self.use_tqdm or not self.verbose) as progressbar:
            for data_batch in self.training_loader:
                batch_losses = self.train_step(data_batch)
                for key in batch_losses:
                    epoch_train_metrics[key].append(batch_losses[key])
                    if self.logger is not None:
                        self.logger.log_batch_value(tag=f"batch_{key}",
                                                    value=batch_losses[key],
                                                    step_ind=self.global_step)

                progressbar.update(1)
                self.global_step += 1
        
        end_time = perf_counter()
        time_taken = end_time - start_time
        if self.verbose:
             print(f"\tTime taken: {time_taken:.4g} seconds")
        return dict(epoch_train_metrics)
    
    def finish_epoch(self,
                     epoch_train_metrics: dict[str, list[torch.Tensor]],
                     epoch_ind: int) -> bool:
        """Bunch of housekeeping after each epoch training loop.
        
        This function:
            - Evaluates on the validation set.
            - Applies learning rate scheduling.
            - Checks for early stopping.
            - Collects train and validation metrics in one place.
            - Optionally writes Tensorboard summaries.

        Parameters:
            epoch_train_metrics: As returned from the last train_epoch call.
            epoch_ind: The index of the epoch (wow).

        Returns:
            Boolean flag from early stopping.
        """
        val_loss_full, val_losses = self.evaluate()

        for key in epoch_train_metrics:
            val_metric = val_losses[key].item()
            train_metric = torch.stack(epoch_train_metrics[key]).mean().item()

            self.full_metrics["val_" + key].append(val_metric)
            self.full_metrics["train_" + key].append(train_metric)
            if self.logger is not None:
                self.logger.log_epoch_values(tag=f"epoch_{key}",
                                             tag_value_dict={"training": train_metric, "validation": val_metric},
                                             epoch_ind=epoch_ind)

        if self.verbose:
            print("\tMetrics:")
            for key in self.full_metrics:
                print(f"\t\t{key}: {self.full_metrics[key][-1]:.6g}")
            print()
        if self.logger is not None:
            self.logger.flush()

    def evaluate(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One evaluation loop.

        Returns:
            - The full evaluation loss (e.g. for early stopping)
            - Dictionary with separate metrics. Note that these are tensors, not np arrays like in the train_model
              function.
        """
        self.model.eval()
        num_batches = len(self.validation_loader)
        val_loss_full = 0.
        val_losses = defaultdict(float)

        # manual progressbar required due to multiprocessing in dataloaders
        with tqdm(total=len(self.validation_loader), desc="Validation", leave=False,
                  disable=not self.use_tqdm or not self.verbose) as progressbar:
            for data_batch in self.validation_loader:
                batch_loss_full, batch_losses = self.eval_step(data_batch)
                
                val_loss_full += batch_loss_full
                for key in batch_losses:
                    val_losses[key] += batch_losses[key]
                progressbar.update(1)
                
        # TODO this is not correct as the last batch may be smaller -> slight bias
        val_loss_full /= num_batches
        for key in val_losses:
            val_losses[key] /= num_batches
        return val_loss_full, dict(val_losses)
    
    def train_step(self,
                   data_batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Standard training step: Get loss (through core_step), backpropagate, apply gradients.
        
        Parameters:
            data_batch: A tuple of images, labels.

        Returns:
            Dictionary mapping loss names to batch averages.
        """
        batch_loss_full, batch_losses = self.core_step(data_batch)
        batch_loss_full.backward()

        if self.logger is not None:
            self.logger.log_gradients(self.model, self.global_step)
            self.logger.log_images(data_batch[0], self.global_step)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return batch_losses
    
    def eval_step(self,
                  data_batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss for validation. No gradients are computed!
        
        Returns:
            The same tuple as train_step, but also an "overall loss" as returned by core step, to be used e.g. for early
            stopping or ReduceLROnPlateau
        """
        with torch.inference_mode():
            batch_loss_full, batch_losses = self.core_step(data_batch)
        return batch_loss_full, batch_losses

    def core_step(self,
                  data_batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Main logic for computing losses. Not implemented as it is model-dependent.
        
        Generally this function should:
        - Split data into inputs/labels
        - Move data to the correct device
        - Apply the model
        - Compute any losses
        - Return the overall loss, and a dictionary with component losses (e.g. for VAE: reconstruction and KLD losses).

        Parameters:
            data_batch: Expected to be a tuple for inputs, labels. Generative modeling doesn't need labels, but we later
                        want to implement conditional models that do. So we just carry the labels around in the dataset.
                        For this reason some of the datasets we use may have "dummy" labels to avoid mismatches.

        Return:
            A tuple. The first entry should be the full loss to be used for training, or things like early stopping.
            Many models will only have one loss, but some have multiple. This will likely be their sum. The second entry
            should be a dictionary mapping names to the individual loss components. Even if you have only one loss in
            the model, just repeat it here.
        """
        raise NotImplementedError
    

class ClassifierTrainer(TrainerBase):
    def __init__(self,
                 label_smoothing: float = 0.,
                 classes: list[str] | None = None,
                 **kwargs):
        """Trainer class for classifiers.

        Parameters:
            label_smoothing: If given a float > 0, apply that amount of label smoothing. Not recommended if you use
                             cutmix/mixup augmentations.
            classes: If given, use these class names for the plot_examples function. Otherwise, classes will just be
                     numbered starting from 0.
        """
        super().__init__(**kwargs)
        self.classes = classes
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def core_step(self,
                  data_batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Standard classifier step. Computes cross-entropy and accuracy."""
        input_batch, label_batch = data_batch
        input_batch = input_batch.to(self.device, non_blocking=True)
        label_batch = label_batch.to(self.device, non_blocking=True)
        logits = self.model(input_batch)
        loss = self.loss_fn(logits, label_batch)
        with torch.inference_mode():
            batch_accuracy = accuracy(logits, label_batch)
        return loss, {"cross_entropy": loss, "accuracy": batch_accuracy}
    
    def plot_examples(self):
        """Show classification example results.
        
        For each image, shows the true class and argmax prediction, with associated probability.
        """
        inputs, y = next(iter(self.validation_loader))
        plot_n_rows = 8  # TODO don't hardcode
        inputs = inputs[:plot_n_rows**2]
        y = y[:plot_n_rows**2]
        with torch.inference_mode():
            probabilities = torch.nn.functional.softmax(self.model(inputs.to(self.device)), dim=1).cpu()
            predictions = probabilities.argmax(axis=1)
        max_class = probabilities.shape[1]
        classes = list(range(max_class)) if self.classes is None else self.classes

        subtitles = []
        for ind in range(inputs.shape[0]):
            pred_here = predictions[ind]
            prob_here = probabilities[ind, pred_here].item()
            true_here = y[ind]
            subtitles.append(f"true: {classes[true_here]} pred: {classes[pred_here]}\nprob: {prob_here:.3f}")
        plot_image_grid(inputs, figure_size=(12, 12), title="Classifications", subtitles=subtitles, n_rows=plot_n_rows)


class TensorboardLogger:
    def __init__(self,
                 logdir: str,
                 step_frequency: int = 100,
                 do_log_batch_values: bool = True,
                 do_log_gradients: bool = True,
                 do_log_images: bool = True):
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        self.step_frequency = step_frequency

        self.do_log_batch_values = do_log_batch_values
        self.do_log_gradients = do_log_gradients
        self.do_log_images = do_log_images

    def log_batch_value(self,
                        tag: str,
                        value: torch.Tensor,
                        step_ind: int):
        if self.do_log_batch_values and not step_ind % self.step_frequency:
            self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step_ind)

    def log_epoch_values(self,
                         tag: str,
                         tag_value_dict: dict[str, torch.Tensor],
                         epoch_ind: int):
        self.writer.add_scalars(main_tag=tag, tag_scalar_dict=tag_value_dict, global_step=epoch_ind)

    def log_gradients(self,
                      model: nn.Module,
                      step_ind: int):
        if self.do_log_gradients and not step_ind % self.step_frequency:
            for name, parameter in model.named_parameters():
                with torch.no_grad():
                    gradient_norm = torch.sqrt((parameter.grad**2).sum())
                self.writer.add_scalar(tag=f"gradient_{name}", scalar_value=gradient_norm, global_step=step_ind)

    def log_images(self,
                   images: torch.Tensor,
                   step_ind: int):
        if self.do_log_images and not step_ind % self.step_frequency:
            self.writer.add_images(tag="input_images", img_tensor=images, global_step=step_ind)

    def flush(self):
        self.writer.flush()
