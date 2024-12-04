import jax.numpy as jnp
import pytest
from flax import nnx
from model.architecture import CrossMultiHeadAttention

def test_cross_multi_head_attention():
    # Initialize parameters
    embed_dim = 16
    seq_len = 4
    batch_size = 2
    num_heads = 4
    
    # Create random input tensors
    x = jnp.ones((batch_size, seq_len, embed_dim))
    kv = jnp.ones((batch_size, seq_len, embed_dim))
    
    # Initialize the module
    attention_module = CrossMultiHeadAttention(embed_dim=embed_dim, rngs=nnx.Rngs(0))
    
    # Forward pass
    output, attention_weights = attention_module(x, kv, num_heads=num_heads)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, embed_dim), "Output shape mismatch"
    
    # Check attention weights shape
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), "Attention weights shape mismatch"
    
    # Optional: Check numerical properties (e.g., attention sum)
    assert jnp.allclose(attention_weights.sum(axis=-1), 1, atol=1e-5), "Attention weights should sum to 1 along the last axis"

# Run the test with pytest
if __name__ == "__main__":
    pytest.main(["-v", __file__])
