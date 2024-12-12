import torch
from torch.utils.data import DataLoader
from data.reverse_task_data import ReverseTaskDataset
from data.utils import custom_collate_fn
from data.copy_task_data import CopyTaskDataset
from tasks.seq2seq_task import Seq2SeqTaskTransformerModel, Seq2SeqTaskEncoderModel
from training.trainer import TrainerWithEarlyStopping
from flax import nnx
import optax
import numpy as np
from checkpoints.checkpoint_manager import save_model
import jax.numpy as jnp
import os

# Configuration
vocab_size = 19
input_dim = vocab_size+1
embed_dim = 128
feedforward_dim = 128
num_blocks = 2
dropout_prob = 0.3
seq_length = 10
target_seq_length = 10
batch_size = 32
num_heads = 8
mask = np.tril(np.ones((target_seq_length, target_seq_length)), k=0)
batch_size = 32
dataset_split = [0.7, 0.1, 0.2]
metrics_history = {'train_loss': []}
num_epochs = 50
model_args = (input_dim, embed_dim, feedforward_dim, num_blocks, dropout_prob, num_heads)
learning_rate = 0.001
tracking_metric = 'val_loss'
patience = 6

# Dataset
dataset = CopyTaskDataset(num_samples=5000, seq_length=seq_length, vocab_size=vocab_size)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, dataset_split)
train_ds = DataLoader(train_set, 
                      batch_size, 
                      shuffle=True, 
                      drop_last=True, 
                      collate_fn=lambda batch: custom_collate_fn(batch, vocab_size))

val_ds = DataLoader(val_set, 
                      batch_size, 
                      shuffle=True, 
                      drop_last=True, 
                      collate_fn=lambda batch: custom_collate_fn(batch, vocab_size))

# Model and optimizer
#model = Seq2SeqTaskTransformerModel(input_dim, embed_dim, feedforward_dim, num_blocks, dropout_prob, num_heads, rngs=nnx.Rngs(0))
model = Seq2SeqTaskEncoderModel(input_dim, embed_dim, feedforward_dim, num_blocks, dropout_prob, num_heads, rngs=nnx.Rngs(0))
ckpt_dir = f'savedModels\{model.model_backbone.__class__.__name__}'
optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

#print(model.model_backbone.__class__.__name__)
# nnx.display(model)
trainer = TrainerWithEarlyStopping(model, mask, optimizer, train_ds, val_ds, 
                                   metrics, num_epochs, tracking_metric, patience, ckpt_dir, mode='min')

metrics_history, best_value = trainer.train_and_evaluate()

#print(metrics_history, best_value)
