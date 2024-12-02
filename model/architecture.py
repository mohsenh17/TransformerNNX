import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from flax import nnx

from model import utils

def scaled_dot_product(
    q: jnp.ndarray, 
    k: jnp.ndarray, 
    v: jnp.ndarray, 
    mask: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape (..., seq_len, d_k).
        k: Key tensor of shape (..., seq_len, d_k).
        v: Value tensor of shape (..., seq_len, d_k).
        mask: Optional mask tensor of shape (..., seq_len, seq_len).
    
    Returns:
        values: Tensor of shape (..., seq_len, d_k) containing the attention-weighted values.
        attention: Tensor of shape (..., seq_len, seq_len) containing attention scores.
    """
    d_k = q.shape[-1]  # Dimensionality of key vectors
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))  # Compute attention logits
    attn_logits = attn_logits / jnp.sqrt(d_k)  # Scale by sqrt(d_k)

    if mask is not None:
        attn_logits = jnp.where(mask == 0, -1e9, attn_logits)  # Apply mask with large negative value

    attention = jax.nn.softmax(attn_logits, axis=-1)  # Softmax over last axis
    values = jnp.matmul(attention, v)  # Compute weighted values
    return values, attention


class MultiHeadAttention(nnx.Module):
    """
    Implements a multi-head attention mechanism.

    Attributes:
    ----------
    qkv_projection : nnx.Linear
        Linear layer for projecting the input into query, key, and value tensors.
    out_projection : nnx.Linear
        Linear layer for projecting the output values back to the embedding dimension.

    Methods:
    -------
    __call__(x, num_heads, mask=None):
        Computes the multi-head attention output for the input tensor `x` with the given number of heads.
    """

    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs) -> None:
        """
        Initializes the MultiHeadAttention module.

        Parameters:
        ----------
        embed_dim : int
            The dimension of the input embeddings.
        rngs : nnx.Rngs
            Random number generators for initializing weights of the projection layers.
        """
        self.qkv_projection = nnx.Linear(embed_dim, 3 * embed_dim, rngs=rngs)
        self.out_projection = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, 
                 x: jnp.ndarray, 
                 num_heads: int, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Applies the multi-head attention mechanism.

        Parameters:
        ----------
        x : jnp.ndarray
            Input tensor of shape `(batch_size, seq_len, embed_dim)`.
        num_heads : int
            The number of attention heads to use.
        mask : Optional[jnp.ndarray], default=None
            Optional mask tensor of shape `(batch_size, seq_len, seq_len)` 
            or `(batch_size, num_heads, seq_len, seq_len)` to apply attention masking.

        Returns:
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            - Output tensor of shape `(batch_size, seq_len, embed_dim)`.
            - Attention weights tensor of shape `(batch_size, num_heads, seq_len, seq_len)`.
        """
        batch_size, seq_len, embed_dim = x.shape
        if mask is not None:
            mask = utils.expand_mask(mask)  # Ensure mask is in the correct 4D format
        
        # Compute query, key, and value projections
        qkv = self.qkv_projection(x)  # Shape: (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, num_heads, -1)  # Split heads
        qkv = qkv.transpose(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_len, d_k)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)  # Split into query, key, value
        
        # Compute scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask)  # Custom function
        
        # Reshape and project output
        values = values.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        out = self.out_projection(values)
        
        return out, attention
