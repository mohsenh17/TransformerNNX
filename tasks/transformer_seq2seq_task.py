import jax.numpy as jnp
from typing import Optional
from flax import nnx
from model.architecture import Transformer

class Seq2SeqTaskModel(nnx.Module):
    """
    A sequence-to-sequence task model using a Transformer architecture.

    This model processes input sequences through an encoder-decoder Transformer 
    for tasks like machine translation or reverse tasks. It outputs logits that can 
    be further processed into predictions.

    Args:
        input_dim (int): Dimension of the input embeddings.
        feedforward_dim (int): Dimension of the hidden feedforward layer in the transformer.
        num_blocks (int): Number of transformer blocks (layers).
        dropout_prob (float): Dropout probability for regularization.
        rngs (nnx.Rngs): Random number generators for model initialization.

    Attributes:
        transformer (Transformer): The underlying Transformer model.
    """
    def __init__(self, 
                 input_dim: int,
                 embed_dim:int, 
                 feedforward_dim: int, 
                 num_blocks: int, 
                 dropout_prob: float, 
                 *, rngs: nnx.Rngs) -> None:
        """
        Initializes the Seq2SeqTaskModel.

        Args:
            input_dim (int): Dimension of the input embeddings.
            feedforward_dim (int): Hidden dimension of the feedforward network.
            num_blocks (int): Number of encoder-decoder layers.
            dropout_prob (float): Dropout probability for Transformer layers.
            rngs (nnx.Rngs): Random number generators for initialization.
        """
        self.embd_projection = nnx.Linear(input_dim, embed_dim, rngs=rngs)
        self.transformer = Transformer(embed_dim, feedforward_dim, num_blocks, dropout_prob, rngs=rngs)
        self.output_projection = nnx.Linear(embed_dim, input_dim, rngs=rngs)

    def __call__(self, 
                 x: jnp.ndarray, 
                 y: jnp.ndarray, 
                 num_heads: int, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass through the Transformer.

        Args:
            x (jnp.ndarray): Input tensor for the encoder, 
                shape `(batch_size, src_seq_length, input_dim)`.
            y (jnp.ndarray): Input tensor for the decoder, 
                shape `(batch_size, tgt_seq_length, input_dim)`.
            num_heads (int): Number of attention heads for the Transformer.
            mask (Optional[jnp.ndarray]): Optional mask tensor, 
                shape `(batch_size, tgt_seq_length, src_seq_length)` 
                or `(batch_size, tgt_seq_length, tgt_seq_length)`.

        Returns:
            jnp.ndarray: Output logits, shape `(batch_size, tgt_seq_length, input_dim)`.
        """
        x = self.embd_projection(x)
        y = self.embd_projection(y)
        out = self.transformer(x, y, num_heads, mask)
        out = self.output_projection(out)
        return out
