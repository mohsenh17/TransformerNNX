import pytest
import jax.numpy as jnp
from flax import nnx
from model.architecture import Transformer

def test_transformer():
    # Parameters
    input_dim = 16
    feedforward_dim = 64
    num_blocks = 2
    seq_len = 10
    batch_size = 2
    num_heads = 4
    dropout_prob = 0.1

    # Input tensor
    x = jnp.ones((batch_size, seq_len, input_dim))

    # Initialize Transformer
    transformer = Transformer(
        input_dim=input_dim,
        feedforward_dim=feedforward_dim,
        num_blocks=num_blocks,
        dropout_prob=dropout_prob,
        rngs=nnx.Rngs(0),
    )

    # Forward pass
    output = transformer(x, num_heads=num_heads)
    assert output.shape == (batch_size, seq_len, input_dim), "Output shape mismatch"

# Run with pytest
if __name__ == "__main__":
    pytest.main(["-v", __file__])