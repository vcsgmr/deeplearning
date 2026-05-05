from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import chain
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
                 early_stopper: EarlyStopping | None = None,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 verbose: bool = True,
                 use_tqdm: bool = False):
        """Base class for training models.

        Any Trainer for a specific kind of model should inherit from this and implement the core_step function.

        Parameters:
            model: The model to train.
            optimizer: torch optimizer instance.
            training_loader, validation_loader: Dataloaders for training/validation sets.
            n_epochs: Number of full iterations over the training loader.
            device: Device on which all the torch stuff should happen (e.g. "cuda").
            logger: Optional TensorboardLogger instance for logging with Tensorboard.
            early_stopper: Optional early stopping object. Pass None to disable early stopping.
            scheduler: Learning rate scheduler. NOTE that this is applied every training STEP, EXCEPT if it's a
                       ReduceLROnPlateau instance, in which case it's applied once after each EPOCH.
            verbose: If True, report on training progress throughout.
            use_tqdm: If True, and verbose is also True, supply per-epoch progress bars.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopper = early_stopper

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
            should_stop = self.finish_epoch(epoch_train_metrics, epoch_ind)
            if should_stop:
                print("Early stopping...")
                break

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

                if (self.scheduler is not None
                    and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                    self.scheduler.step()
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
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss_full)
                
        if self.early_stopper is not None:
            should_stop = self.early_stopper.update(val_loss_full)
        else:
            should_stop = False

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
            if self.scheduler is not None:
                print(f"\tLR is now {self.scheduler.get_last_lr()[0]:.10g}")
            print()
        if self.logger is not None:
            self.logger.flush()
        return should_stop

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
        plot_image_grid(inputs, figure_size=(12, 12), title="Classifications", subtitles=subtitles, n_rows=plot_n_rows,
                        tight_layout=True)


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


class ParameterTracker:
    def __init__(self,
                 model: nn.Module,
                 trainable_only: bool = False,
                 include_buffers: bool = True):
        """Base class for parameter trackers/storages like early stopping or Polyak averaging.

        Parameters:
            model: Model to track.
            trainable_only: If True, we only create EMAs for trainable parameters. Otherwise, anything in the model
                            state dict will be affected.
            include_buffers: If True, also create EMAs for buffers (like Batchnorm moving averages). Only float-type
                             buffers are used!
        """
        self.model = model
        self.trainable_only = trainable_only
        self.include_buffers = include_buffers
        self.tracked_parameters = [param.detach().clone() for param in self.get_parameters()]
        self.backup = []

    def get_parameters(self) -> Iterable[torch.Tensor]:
        """Return all desired parameters."""
        if not self.include_buffers:
            return iter(param for param in self.model.parameters() if param.requires_grad or not self.trainable_only)
        return iter(param for param in chain(self.model.parameters(), self.model.buffers())
                    if (param.requires_grad or not self.trainable_only) and torch.is_floating_point(param))
    
    @torch.no_grad()
    def apply_parameters(self):
        """Overwrite model parameters with EMA parameters while also creating a backup."""
        self.make_backup()
        for tracked_param, model_param in zip(self.tracked_parameters, self.get_parameters()):
            model_param.copy_(tracked_param)

    def make_backup(self):
        """Make a backup of original model parameters."""
        if not self.backup:
            print("Creating backup...")
            self.backup = [param.detach().clone() for param in self.get_parameters()]
        else:
            print("backup has been created already! This backup has been SKIPPED.")

    @torch.no_grad()
    def apply_backup(self):
        """Restore backed up model parameters."""
        for backup_param, model_param in zip(self.backup, self.get_parameters()):
            model_param.copy_(backup_param)


class EarlyStopping(ParameterTracker):
    def __init__(self,
                 model: nn.Module,
                 patience: int,
                 direction: str = "min",
                 min_delta: float = 0.0001,
                 verbose: bool = False,
                 trainable_only: bool = False,
                 include_buffers: bool = True,
                 restore_best: bool = False):
        """Stop training if target metric does not improve.

        The model parameters with best performance are tracked and restored at the end.

        Parameters:
            model: Model to track.
            patience: How many iterations without improvement to tolerate. For example, patience=2 means that two
                      iterations *in a row* without improvement are okay; stopping would be triggered after the third
                      iteration in arow without improvement.
            direction: Whether the metric of interest is minimized (e.g. loss) or maximized (e.g. accuracy).
            min_delta: An improvement is only counted as such if it is better by at least this amount.
            verbose: If True, report on how things are going.
            trainable_only: See notes in base class.
            include_buffers: See notes in base class.
            restore_best: If True, when the stop signal is triggered, the best parameters are restored to the model.
                          Otherwise, you have to do this manually later.
        """
        super().__init__(model, trainable_only, include_buffers)
        if direction not in ["min", "max"]:
            raise ValueError(f"direction should be 'min' or 'max', you passed {direction}")
        self.best_value = np.inf if direction == "min" else -np.inf
        self.direction = direction
        self.min_delta = min_delta

        self.patience = patience
        self.disappointment = 0
        self.verbose = verbose
        if verbose and patience is None:
            print("EarlyStopping with patience None -- noop and will never stop")
        self.restore_best = restore_best

    def update(self,
               value: torch.Tensor) -> bool:
        """Run one 'iteration' of early stopping.

        This updates the patience parameter, and if stopping is triggered, sends a signal to stop training. This
        function does *not* actually stop the training process; this needs to be handled in the training function
        based on the bool this function returns. *Optionally* restores the best tracked model parameters.

        Parameters:
            value: New value to compare to best so far.
        """
        if self.patience is None:
            return False
            
        if ((self.direction == "min" and value < self.best_value - self.min_delta) 
            or (self.direction == "max" and value > self.best_value + self.min_delta)):
            self.best_value = value
            self.update_best()
            self.disappointment = 0
            if self.verbose:
                print("New best value found; no longer disappointed")
            return False
        # else
        self.disappointment += 1
        if self.verbose:
            print(f"EarlyStopping disappointment increased to {self.disappointment}")

        if self.disappointment > self.patience:
            if self.verbose:
                print("EarlyStopping has become too disappointed; now would be a good time to cancel training")
                if self.restore_best:
                    print("Restoring best model from state_dict")
                    self.apply_parameters()
            return True
        #else
        return False
            
    @torch.no_grad()
    def update_best(self):
        """Update saved state with new best."""
        for best_param, model_param in zip(self.tracked_parameters, self.get_parameters()):
            best_param.copy_(model_param)
