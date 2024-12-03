import pytest
import jax.numpy as jnp
from flax import nnx
from model.architecture import TransformerEncoder

def test_transformer_encoder_initialization():
    """
    Test that the TransformerEncoder initializes correctly.
    """
    input_dim = 64
    feedforward_dim = 128
    num_blocks = 2
    dropout_prob = 0.1
    encoder = TransformerEncoder(input_dim, feedforward_dim, num_blocks, dropout_prob, rngs=nnx.Rngs(0))
    
    assert len(encoder.blocks) == num_blocks, "Incorrect number of encoder blocks initialized"


def test_transformer_encoder_output_shape():
    """
    Test that the TransformerEncoder produces outputs with the correct shape.
    """
    batch_size, seq_len, input_dim = 2, 10, 64
    feedforward_dim = 128
    num_blocks = 2
    dropout_prob = 0.1
    encoder = TransformerEncoder(input_dim, feedforward_dim, num_blocks, dropout_prob, rngs=nnx.Rngs(0))

    x = jnp.ones((batch_size, seq_len, input_dim))
    out = encoder(x)
    assert out.shape == (batch_size, seq_len, input_dim), "Output shape is incorrect"


def test_transformer_encoder_attention_weights():
    """
    Test that the TransformerEncoder correctly computes attention weights.
    """
    batch_size, seq_len, input_dim = 2, 10, 64
    feedforward_dim = 128
    num_blocks = 2
    dropout_prob = 0.1
    encoder = TransformerEncoder(input_dim, feedforward_dim, num_blocks, dropout_prob, rngs=nnx.Rngs(0))

    x = jnp.ones((batch_size, seq_len, input_dim))
    attention_weights = encoder.get_attention_weights(x)
    assert len(attention_weights) == num_blocks, "Incorrect number of attention weights returned"
    for weight in attention_weights:
        assert weight.shape == (batch_size, 8, seq_len, seq_len), "Attention weight shape is incorrect"
