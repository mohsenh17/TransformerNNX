import os
import jax
import numpy as np
from training.transformer_utils import train_step, eval_step
from checkpoints.checkpoint_manager import save_model
from collections import defaultdict
class TrainerWithEarlyStopping:
    def __init__(self, model, mask, optimizer, train_ds, val_ds, 
                 metrics, num_epochs, tracking_metric, patience, ckpt_dir, mode='min'):
        """
        Initializes the trainer with necessary parameters.
        
        Args:
            model: The model to train and evaluate.
            optimizer: Optimizer for training the model.
            train_ds: Training dataset.
            val_ds: Validation dataset.
            metrics: Metrics object for tracking performance.
            num_epochs (int): Maximum number of epochs to train.
            tracking_metric (str): Metric to track for early stopping.
            patience (int): Number of epochs to wait without improvement before stopping.
            manager: Object managing checkpointing and state saving.
            mode (str): 'min' for metrics to minimize, 'max' for metrics to maximize.
        """
        self.model = model
        self.mask = mask
        self.optimizer = optimizer
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.tracking_metric = tracking_metric
        self.patience = patience
        self.mode = mode
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.metrics_history = defaultdict(list)
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0

    def update_metrics_history(self, prefix):
        """
        Updates the metrics history with current metrics.
        
        Args:
            prefix (str): Prefix for metric keys (e.g., 'train', 'val').
        """
        for metric, value in self.metrics.compute().items():
            self.metrics_history[f'{prefix}_{metric}'].append(value)

    def log_metrics(self, epoch):
        """
        Logs metrics dynamically for the given epoch.
        
        Args:
            epoch (int): Current epoch number.
        """
        print(f"Epoch: {epoch + 1}/{self.num_epochs}, ", end="")
        
        # Prepare the metric strings dynamically
        metric_strings = []
        for key, values in self.metrics_history.items():
            # Ensure the metric has at least one value
            if values:
                metric_strings.append(f"{key}: {values[-1]:.4f}")
        
        # Print the metrics
        print(", ".join(metric_strings))

    def train_and_evaluate(self):
        """
        Trains and evaluates the model, implementing early stopping and model saving.

        Args:
            trainer: Trainer object with `train_step` and `eval_step` methods.

        Returns:
            dict: Metrics history.
            float: Best value for the tracking metric.
            any: Final PRNG key value after training.
        """
        for epoch in range(self.num_epochs):
            # Training phase
            for batch in self.train_ds:
                train_step(self.model, self.optimizer, self.metrics, batch, self.mask)
            self.update_metrics_history('train')
            self.metrics.reset()
            #self.log_metrics(epoch, 'train')

            # Validation phase
            for batch in self.val_ds:
                eval_step(self.model, self.metrics, batch, self.mask)
            self.update_metrics_history('val')
            self.metrics.reset()
            self.log_metrics(epoch)

            # Early stopping and model saving
            current_value = self.metrics_history[f'{self.tracking_metric}'][-1]
            if (self.mode == 'min' and current_value < self.best_value) or (self.mode == 'max' and current_value > self.best_value):
                self.best_value = current_value
                self.best_epoch = epoch
                print(f"\t Best {self.tracking_metric} {self.best_value:.4f}. Saving model...")
                save_model(self.model, self.ckpt_dir)
            elif epoch - self.best_epoch >= self.patience:
                print("Early stopping triggered.")
                break

        return self.metrics_history, self.best_value
