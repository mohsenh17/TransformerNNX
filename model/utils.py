import jax
import jax.numpy as jnp
from typing import Optional, Tuple


def expand_mask(mask: jnp.ndarray) -> jnp.ndarray:
    """
    Expands a given mask to ensure it has 4 dimensions:
    - If the mask is 2D: it will be broadcasted over both batch size and number of heads.
    - If the mask is 3D: it will be broadcasted over the number of heads.
    - If the mask is already 4D, it will be returned as-is.

    Parameters:
    ----------
    mask : jnp.ndarray
        The input mask, which can be 2D, 3D, or 4D. The mask is expected to be of shape:
        - 2D: (seq_length, seq_length)
        - 3D: (batch_size, seq_length, seq_length)
        - 4D: (batch_size, num_heads, seq_length, seq_length)

    Returns:
    -------
    jnp.ndarray
        The mask with 4 dimensions. The output shape depends on the input shape:
        - 2D input: shape will be (1, 1, seq_length, seq_length)
        - 3D input: shape will be (batch_size, 1, seq_length, seq_length)
        - 4D input: shape will be (batch_size, num_heads, seq_length, seq_length)
    """
    
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    
    if mask.ndim == 2:
        # 2D mask, needs to be broadcasted over batch size and number of heads
        mask = jnp.expand_dims(mask, axis=0)  # Add batch dimension (0)
        mask = jnp.expand_dims(mask, axis=0)  # Add heads dimension (1)
    
    elif mask.ndim == 3:
        # 3D mask, needs to be broadcasted over number of heads
        mask = jnp.expand_dims(mask, axis=1)  # Add heads dimension (1)
    
    # If the mask is already 4D, leave it as is (batch_size, num_heads, seq_len, seq_len)
    
    return mask
