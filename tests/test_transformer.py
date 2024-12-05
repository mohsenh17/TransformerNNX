import pytest
import jax.numpy as jnp
from flax import nnx
from model.architecture import Transformer

@pytest.fixture
def transformer():
    """Fixture to create a Transformer instance for testing."""
    input_dim = 32
    feedforward_dim = 64
    num_blocks = 2
    dropout_prob = 0.1
    rng = nnx.Rngs(seed=42)  # Fixed random seed for reproducibility
    return Transformer(
        input_dim=input_dim,
        feedforward_dim=feedforward_dim,
        num_blocks=num_blocks,
        dropout_prob=dropout_prob,
        rngs=rng,
    )
def test_transformer_initialization(transformer):
    """Test that the Transformer initializes correctly."""
    assert transformer.encoder is not None, "Encoder should be initialized"
    assert transformer.decoder is not None, "Decoder should be initialized"
    assert transformer.out_projection is not None, "Output projection should be initialized"


def test_transformer_forward_pass(transformer):
    """Test the Transformer forward pass."""
    batch_size = 16
    src_seq_len = 10
    tgt_seq_len = 10
    input_dim = 32

    # Create dummy inputs
    x = jnp.ones((batch_size, src_seq_len, input_dim))  # Source sequence
    y = jnp.ones((batch_size, tgt_seq_len, input_dim))  # Target sequence
    num_heads = 4

    # Forward pass
    output = transformer(x, y, num_heads)

    # Check output shape
    assert output.shape == (batch_size, tgt_seq_len, input_dim), (
        f"Expected output shape {(batch_size, tgt_seq_len, input_dim)}, got {output.shape}"
    )


def test_transformer_with_mask(transformer):
    """Test the Transformer with a mask."""
    batch_size = 16
    src_seq_len = 10
    tgt_seq_len = 10
    input_dim = 32

    # Create dummy inputs
    x = jnp.ones((batch_size, src_seq_len, input_dim))  # Source sequence
    y = jnp.ones((batch_size, tgt_seq_len, input_dim))  # Target sequence
    mask = jnp.ones((src_seq_len, tgt_seq_len))  # Example mask
    num_heads = 4

    # Forward pass
    output = transformer(x, y, num_heads, mask=mask)

    # Check output shape
    assert output.shape == (batch_size, tgt_seq_len, input_dim), (
        f"Expected output shape {(batch_size, tgt_seq_len, input_dim)}, got {output.shape}"
    )

# Run with pytest
if __name__ == "__main__":
    pytest.main(["-v", __file__])