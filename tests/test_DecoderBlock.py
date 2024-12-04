import pytest
import jax.numpy as jnp
from flax import nnx
from model.architecture import DecoderBlock

def test_decoder_block():
    # Parameters
    input_dim = 16
    feedforward_dim = 64
    seq_len = 10
    batch_size = 2
    num_heads = 4
    dropout_prob = 0.1

    # Input tensors
    x = jnp.ones((batch_size, seq_len, input_dim))  # Decoder input
    encoder_kv = jnp.ones((batch_size, seq_len, input_dim))  # Encoder output

    # Initialize the DecoderBlock
    decoder_block = DecoderBlock(
        input_dim=input_dim,
        feedforward_dim=feedforward_dim,
        dropout_prob=dropout_prob,
        rngs=nnx.Rngs(0),
    )

    # Forward pass
    output = decoder_block(x, encoder_kv, num_heads=num_heads)

    # Check output shape
    assert output.shape == (batch_size, seq_len, input_dim), "Output shape mismatch"

# Run with pytest
if __name__ == "__main__":
    pytest.main(["-v", __file__])
