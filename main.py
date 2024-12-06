import torch
from torch.utils.data import DataLoader
from data.reverse_task_data import ReverseTaskDataset, custom_collate_fn
from tasks.reverse_task import ReverseTaskModel
from training.trainer import train_step
from flax import nnx
import optax
import numpy as np

# Configuration
vocab_size = 19
seq_length = 10
target_seq_length = 10
batch_size = 32
mask = np.tril(np.ones((target_seq_length, target_seq_length)), k=0)

# Dataset
dataset = ReverseTaskDataset(num_samples=1000, seq_length=seq_length, vocab_size=vocab_size)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.1, 0.2])
train_ds = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True, collate_fn=lambda batch: custom_collate_fn(batch, vocab_size))

# Model and optimizer
model = ReverseTaskModel(20, 36, 2, 0.1, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(0.01))
metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

# Training loop
metrics_history = {'train_loss': []}
num_epochs = 40
for epoch in range(num_epochs):
    for batch in train_ds:
        loss, logits = train_step(model, optimizer, metrics, batch, mask)
    for metric, value in metrics.compute().items():
        metrics_history[f'train_{metric}'].append(value)
    metrics.reset()
    print(f"[train] epoch: {epoch + 1}/{num_epochs}, loss: {metrics_history['train_loss'][-1]:.4f}")
