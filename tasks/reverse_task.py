import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from flax import nnx
import optax
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model.architecture import Transformer
from data.reverse_task_data import ReverseTaskDataset

from flax import nnx
import jax.numpy as jnp
from typing import Optional

class ReverseTaskModel(nnx.Module):
    """
    A model that uses a Transformer architecture for a reverse task (such as sequence-to-sequence tasks). 
    This model accepts inputs and produces outputs via a Transformer model.

    Args:
        input_dim (int): The input dimension size (usually the embedding dimension).
        feedforward_dim (int): The hidden dimension of the feedforward network in the transformer.
        num_blocks (int): The number of transformer blocks (layers).
        dropout_prob (float): The probability of dropout in the transformer layers.
        rngs (nnx.Rngs): Random number generators used for initializing the model.

    Attributes:
        transformer (Transformer): A Transformer model that processes the input data.
    """

    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int, 
                 num_blocks: int, 
                 dropout_prob: float, 
                 *, rngs: nnx.Rngs) -> None:
        """
        Initializes the ReverseTaskModel by creating a Transformer instance.

        Args:
            input_dim (int): The input dimension size (embedding dimension).
            feedforward_dim (int): The feedforward network hidden dimension.
            num_blocks (int): Number of transformer layers.
            dropout_prob (float): Dropout probability.
            rngs (nnx.Rngs): Random number generator states for model initialization.
        """
        self.transformer = Transformer(input_dim, feedforward_dim, num_blocks, dropout_prob, rngs=rngs)

    def __call__(self, 
                 x: jnp.ndarray, 
                 y: jnp.ndarray, 
                 num_heads: int, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass through the model.

        Args:
            x (jnp.ndarray): The input to the encoder (of shape `(batch_size, seq_length, input_dim)`).
            y (jnp.ndarray): The input to the decoder (of shape `(batch_size, seq_length, input_dim)`).
            num_heads (int): The number of attention heads to use in the transformer.
            mask (Optional[jnp.ndarray]): An optional mask (of shape `(batch_size, seq_length, seq_length)`).

        Returns:
            jnp.ndarray: The output logits of the model (of shape `(batch_size, seq_length, input_dim)`).
        """
        return self.transformer(x, y, num_heads, mask)
