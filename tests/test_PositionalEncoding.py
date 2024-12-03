import jax.numpy as jnp
from model.architecture import PositionalEncoding
from flax import nnx
import pytest
import matplotlib.pyplot as plt


@pytest.fixture
def positional_encoding():
    # Initialize the PositionalEncoding instance
    return PositionalEncoding(d_model=64, max_seq_len=10, rngs=nnx.Rngs(0))


def test_positional_encoding_shape(positional_encoding):
    """
    Test the shape of the positional encoding.
    """
    # Input tensor with batch size 2, sequence length 10, and embedding dimension 64
    x = jnp.ones((2, 10, 64))
    
    # Get the positional encodings added to the input tensor
    output = positional_encoding(x)
    
    # Check if the output shape is the same as the input shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, but got {output.shape}"


def test_positional_encoding_visualization(positional_encoding):
    """
    Test the positional encoding and visualize the pattern.
    """
    # Generate the positional encoding for a given input tensor (batch size 1, seq_len 10, d_model 64)
    x = jnp.ones((1, 10, 64))
    output = positional_encoding(x)
    
    # Extract the positional encoding part (first element of the batch)
    pe = positional_encoding.pe[0]  # (max_seq_len, d_model)
    
    # Plot the positional encodings
    plt.figure(figsize=(10, 6))
    plt.imshow(pe, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Positional Encodings")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.show()


if __name__ == "__main__":
    pytest.main()
