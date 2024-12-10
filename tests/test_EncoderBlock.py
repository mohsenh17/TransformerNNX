import pytest
import jax.numpy as jnp
from flax import nnx
from model.architecture import EncoderBlock

def test_encoder_block_initialization():
    """
    Test that the EncoderBlock initializes without errors.
    """
    input_dim = 64
    feedforward_dim = 128
    dropout_prob = 0.1
    num_heads = 8
    encoder = EncoderBlock(input_dim, feedforward_dim, dropout_prob, num_heads, rngs=nnx.Rngs(0))
    
    assert encoder.mha is not None
    assert len(encoder.linear) == 3
    assert encoder.norm1 is not None
    assert encoder.norm2 is not None
    assert encoder.dropout is not None


def test_encoder_block_output_shape():
    """
    Test that the EncoderBlock produces outputs with the correct shape.
    """
    batch_size, seq_len, input_dim = 2, 10, 64
    feedforward_dim = 128
    dropout_prob = 0.1
    num_heads = 8
    encoder = EncoderBlock(input_dim, feedforward_dim, dropout_prob, num_heads, rngs=nnx.Rngs(0))

    x = jnp.ones((batch_size, seq_len, input_dim))
    out = encoder(x)
    assert out.shape == (batch_size, seq_len, input_dim), "Output shape is incorrect"


def test_encoder_block_with_mask():
    """
    Test that the EncoderBlock correctly handles attention masks.
    """
    batch_size, seq_len, input_dim = 2, 10, 64
    feedforward_dim = 128
    dropout_prob = 0.1
    num_heads = 8
    encoder = EncoderBlock(input_dim, feedforward_dim, dropout_prob, num_heads, rngs=nnx.Rngs(0))

    x = jnp.ones((batch_size, seq_len, input_dim))
    mask = jnp.ones((seq_len, seq_len))  # Simple 2D mask
    out = encoder(x, mask=mask)
    assert out.shape == (batch_size, seq_len, input_dim), "Output shape with mask is incorrect"


def test_encoder_block_invalid_input():
    """
    Test that the EncoderBlock raises errors for invalid inputs.
    """
    input_dim = 64
    feedforward_dim = 128
    dropout_prob = 0.1
    num_heads = 8
    encoder = EncoderBlock(input_dim, feedforward_dim, dropout_prob, num_heads, rngs=nnx.Rngs(0))

    # Invalid input shape
    with pytest.raises(ValueError):
        x = jnp.ones((10, input_dim))  # Missing batch dimension
        encoder(x)

    # Invalid mask shape
    with pytest.raises(ValueError):
        x = jnp.ones((2, 10, input_dim))
        mask = jnp.ones((5, 5))  # Mismatched sequence length
        encoder(x, mask=mask)
