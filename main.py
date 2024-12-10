import torch
from torch.utils.data import DataLoader
from data.reverse_task_data import ReverseTaskDataset
from data.utils import custom_collate_fn
from data.copy_task_data import CopyTaskDataset
from tasks.transformer_seq2seq_task import Seq2SeqTaskModel
from training.trainer import train_step, pred_step
from flax import nnx
import optax
import numpy as np

import jax.numpy as jnp

# Configuration
vocab_size = 19
input_dim = vocab_size+1
embed_dim = 512
feedforward_dim = 128
num_blocks = 2
dropout_prob = 0.1
seq_length = 10
target_seq_length = 10
batch_size = 32
num_heads = 8
mask = np.tril(np.ones((target_seq_length, target_seq_length)), k=0)
batch_size = 32
dataset_split = [0.7, 0.1, 0.2]
metrics_history = {'train_loss': []}
num_epochs = 250

# Dataset
dataset = CopyTaskDataset(num_samples=1000, seq_length=seq_length, vocab_size=vocab_size)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, dataset_split)
train_ds = DataLoader(train_set, 
                      batch_size, 
                      shuffle=True, 
                      drop_last=True, 
                      collate_fn=lambda batch: custom_collate_fn(batch, vocab_size))

# Model and optimizer
model = Seq2SeqTaskModel(input_dim, embed_dim, 
                         feedforward_dim, num_blocks, 
                         dropout_prob, num_heads, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(0.001))
metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

# Training loop
for epoch in range(num_epochs):
    for batch in train_ds:
        loss, logits = train_step(model, optimizer, metrics, batch, mask)
    for metric, value in metrics.compute().items():
        metrics_history[f'train_{metric}'].append(value)
    metrics.reset()
    print(f"[train] epoch: {epoch + 1}/{num_epochs}, loss: {metrics_history['train_loss'][-1]:.4f}")
    
    #print("logits:", logits.argmax(axis=-1)[0])
    #print("labels:", np.argmax(batch['targets'], axis=-1)[0])

# Testing loop
test_ds = DataLoader(test_set, 
                     batch_size, 
                     shuffle=True, 
                     drop_last=True, 
                     collate_fn=lambda batch: custom_collate_fn(batch, vocab_size))
all_preds = []
all_labels = []
for batch in test_ds:
    all_labels.append(np.argmax(batch['targets'], axis=-1))
    preds = pred_step(model, batch, target_seq_length)
    print("preds:", preds[0])
    print("labels:", np.argmax(batch['targets'], axis=-1)[0])
    all_preds.append(preds)
    break
