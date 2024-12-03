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

class EncoderBlock(nnx.Module):
    """
    A single Transformer encoder block consisting of multi-head attention, feedforward layers, 
    layer normalization, and dropout.

    Attributes:
        mha (MultiHeadAttention): Multi-head attention mechanism.
        linear (list[nnx.Module]): A list of feedforward layers including two linear transformations and dropout.
        norm1 (nnx.LayerNorm): Layer normalization after the multi-head attention layer.
        norm2 (nnx.LayerNorm): Layer normalization after the feedforward layers.
        dropout (nnx.Dropout): Dropout layer applied after the multi-head attention and feedforward layers.
    
    Args:
        input_dim (int): Dimensionality of the input embeddings.
        feedforward_dim (int): Dimensionality of the intermediate feedforward layer.
        dropout_prob (float): Probability of dropout.
        rngs (nnx.Rngs): Random number generators for reproducibility.
    """

    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int, 
                 dropout_prob: float, 
                 *, rngs: nnx.Rngs):
        self.mha = MultiHeadAttention(embed_dim=input_dim, rngs=rngs)
        self.linear = [
            nnx.Linear(input_dim, feedforward_dim, rngs=rngs),
            nnx.Dropout(dropout_prob, rngs=rngs),
            nnx.Linear(feedforward_dim, input_dim, rngs=rngs),
        ]
        self.norm1 = nnx.LayerNorm(input_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(input_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)

    def __call__(self, 
                 x: jnp.ndarray, 
                 num_heads: int = 8, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass for the Transformer encoder block.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, input_dim).
            num_heads (int): Number of attention heads in the multi-head attention mechanism. Default is 8.
            mask (Optional[jnp.ndarray]): Optional attention mask of shape 
                                          (seq_len, seq_len), 
                                          (batch_size, seq_len, seq_len), 
                                          or (batch_size, num_heads, seq_len, seq_len).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, seq_len, input_dim).
        """
        # Multi-Head Attention with residual connection and layer norm
        mha_out, _ = self.mha(x, num_heads=num_heads, mask=mask)
        x = x + self.dropout(mha_out)
        x = self.norm1(x)
        
        # Feedforward network with residual connection and layer norm
        for l in self.linear:
            x = l(x)
        x = x + self.dropout(x)
        x = self.norm2(x)
        return x
