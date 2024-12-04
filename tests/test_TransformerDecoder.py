import pytest
import jax.numpy as jnp
from flax import nnx
from model.architecture import TransformerDecoder

def test_transformer_decoder():
    # Parameters
    input_dim = 16
    feedforward_dim = 64
    num_blocks = 2
    seq_len = 10
    batch_size = 2
    num_heads = 4
    dropout_prob = 0.1

    # Input tensors
    x = jnp.ones((batch_size, seq_len, input_dim))  # Decoder input
    encoder_kv = jnp.ones((batch_size, seq_len, input_dim))  # Encoder output

    # Initialize the TransformerDecoder
    decoder = TransformerDecoder(
        input_dim=input_dim,
        feedforward_dim=feedforward_dim,
        num_blocks=num_blocks,
        dropout_prob=dropout_prob,
        rngs=nnx.Rngs(0),
    )

    # Forward pass
    output = decoder(x, encoder_kv, num_heads=num_heads)
    assert output.shape == (batch_size, seq_len, input_dim), "Output shape mismatch"

    # Test self-attention weights
    mha_weights = decoder.get_mha_attention_weights(x, num_heads=num_heads)
    assert len(mha_weights) == num_blocks, "Mismatch in number of self-attention weights"

    # Test cross-attention weights
    cmha_weights = decoder.get_cmha_attention_weights(x, encoder_kv, num_heads=num_heads)
    assert len(cmha_weights) == num_blocks, "Mismatch in number of cross-attention weights"

# Run with pytest
if __name__ == "__main__":
    pytest.main(["-v", __file__])
