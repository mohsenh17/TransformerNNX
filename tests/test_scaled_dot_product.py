import jax
import jax.numpy as jnp
from model.architecture import scaled_dot_product


def test_scaled_dot_product():
    # Initialize example input dimensions
    seq_len, d_k = 3, 2
    rng = jax.random.PRNGKey(0)  # Initialize random number generator
    rng, rand1 = jax.random.split(rng)

    # Generate random q, k, v tensors
    qkv = jax.random.normal(rand1, (3, seq_len, d_k))
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Call the scaled_dot_product function
    values, attention = scaled_dot_product(q, k, v)

    # Print outputs for manual inspection
    print("Q\n", q)
    print("K\n", k)
    print("V\n", v)
    print("Values\n", values)
    print("Attention\n", attention)

    # Assert basic properties of outputs
    assert values.shape == (seq_len, d_k), "Values have incorrect shape."
    assert attention.shape == (seq_len, seq_len), "Attention has incorrect shape."
    assert jnp.allclose(jnp.sum(attention, axis=-1), 1), "Attention rows must sum to 1."

#python -m tests.test_scaled_dot_product

if __name__ == "__main__":
    test_scaled_dot_product()
