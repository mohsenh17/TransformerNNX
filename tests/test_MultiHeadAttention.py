import jax.numpy as jnp
import pytest
from flax import nnx
from model.architecture import scaled_dot_product, MultiHeadAttention

def test_multi_head_attention_initialization():
    """
    Test that the MultiHeadAttention module initializes without errors.
    """
    embed_dim = 64
    mha = MultiHeadAttention(embed_dim, rngs=nnx.Rngs(0))
    assert mha.qkv_projection is not None
    assert mha.out_projection is not None


def test_multi_head_attention_output_shape():
    """
    Test that the MultiHeadAttention module produces outputs with the correct shapes.
    """
    batch_size, seq_len, embed_dim = 2, 10, 64
    num_heads = 4
    x = jnp.ones((batch_size, seq_len, embed_dim))
    mha = MultiHeadAttention(embed_dim, rngs=nnx.Rngs(0))
    out, attention = mha(x, num_heads)

    # Assert output shapes
    assert out.shape == (batch_size, seq_len, embed_dim), "Output shape is incorrect"
    assert attention.shape == (batch_size, num_heads, seq_len, seq_len), "Attention shape is incorrect"


@pytest.mark.parametrize(
    "mask_shape",
    [
        (10, 10),  # 2D mask
        (2, 10, 10),  # 3D mask
        (2, 4, 10, 10),  # 4D mask
    ],
)
def test_multi_head_attention_with_mask(mask_shape):
    """
    Test that MultiHeadAttention correctly handles different mask shapes.
    """
    batch_size, seq_len, embed_dim = 2, 10, 64
    num_heads = 4
    x = jnp.ones((batch_size, seq_len, embed_dim))
    mask = jnp.ones(mask_shape)
    mha = MultiHeadAttention(embed_dim, rngs=nnx.Rngs(0))
    out, attention = mha(x, num_heads, mask)

    # Assert output shapes
    assert out.shape == (batch_size, seq_len, embed_dim), "Output shape is incorrect with mask"
    assert attention.shape == (batch_size, num_heads, seq_len, seq_len), "Attention shape is incorrect with mask"


def test_multi_head_attention_invalid_input():
    """
    Test that MultiHeadAttention raises errors for invalid inputs.
    """
    embed_dim = 64
    mha = MultiHeadAttention(embed_dim, rngs=nnx.Rngs(0))

    # Test for invalid input shape
    with pytest.raises(ValueError):
        x = jnp.ones((2, 10))  # Missing embedding dimension
        mha(x, num_heads=4)

    # Test for mismatched mask shape
    with pytest.raises(ValueError):
        x = jnp.ones((2, 10, embed_dim))
        mask = jnp.ones((5, 5))  # Mismatched sequence length
        mha(x, num_heads=4, mask=mask)