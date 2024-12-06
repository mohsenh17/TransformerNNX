import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from flax import nnx

from model import utils

def scaled_dot_product(
    q: jnp.ndarray, 
    k: jnp.ndarray, 
    v: jnp.ndarray, 
    mask: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape (..., seq_len, d_k).
        k: Key tensor of shape (..., seq_len, d_k).
        v: Value tensor of shape (..., seq_len, d_k).
        mask: Optional mask tensor of shape (..., seq_len, seq_len).
    
    Returns:
        values: Tensor of shape (..., seq_len, d_k) containing the attention-weighted values.
        attention: Tensor of shape (..., seq_len, seq_len) containing attention scores.
    """
    d_k = q.shape[-1]  # Dimensionality of key vectors
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))  # Compute attention logits
    attn_logits = attn_logits / jnp.sqrt(d_k)  # Scale by sqrt(d_k)

    if mask is not None:
        attn_logits = jnp.where(mask == 0, -1e9, attn_logits)  # Apply mask with large negative value

    attention = jax.nn.softmax(attn_logits, axis=-1)  # Softmax over last axis
    values = jnp.matmul(attention, v)  # Compute weighted values
    return values, attention


class MultiHeadAttention(nnx.Module):
    """
    Implements a multi-head attention mechanism.

    Attributes:
    ----------
    qkv_projection : nnx.Linear
        Linear layer for projecting the input into query, key, and value tensors.
    out_projection : nnx.Linear
        Linear layer for projecting the output values back to the embedding dimension.

    Methods:
    -------
    __call__(x, num_heads, mask=None):
        Computes the multi-head attention output for the input tensor `x` with the given number of heads.
    """

    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs) -> None:
        """
        Initializes the MultiHeadAttention module.

        Parameters:
        ----------
        embed_dim : int
            The dimension of the input embeddings.
        rngs : nnx.Rngs
            Random number generators for initializing weights of the projection layers.
        """
        self.qkv_projection = nnx.Linear(embed_dim, 3 * embed_dim, rngs=rngs)
        self.out_projection = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, 
                 x: jnp.ndarray, 
                 num_heads: int, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Applies the multi-head attention mechanism.

        Parameters:
        ----------
        x : jnp.ndarray
            Input tensor of shape `(batch_size, seq_len, embed_dim)`.
        num_heads : int
            The number of attention heads to use.
        mask : Optional[jnp.ndarray], default=None
            Optional mask tensor of shape `(batch_size, seq_len, seq_len)` 
            or `(batch_size, num_heads, seq_len, seq_len)` to apply attention masking.

        Returns:
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            - Output tensor of shape `(batch_size, seq_len, embed_dim)`.
            - Attention weights tensor of shape `(batch_size, num_heads, seq_len, seq_len)`.
        """
        batch_size, seq_len, embed_dim = x.shape
        if mask is not None:
            mask = utils.expand_mask(mask)  # Ensure mask is in the correct 4D format
        
        # Compute query, key, and value projections
        qkv = self.qkv_projection(x)  # Shape: (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, num_heads, -1)  # Split heads
        qkv = qkv.transpose(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_len, d_k)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)  # Split into query, key, value
        
        # Compute scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask)  # Custom function
        
        # Reshape and project output
        values = values.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        out = self.out_projection(values)
        
        return out, attention

class EncoderBlock(nnx.Module):
    """
    A single Transformer encoder block consisting of multi-head attention, feedforward layers, 
    layer normalization, and dropout.

    Attributes:
        mha (MultiHeadAttention): Multi-head attention mechanism.
        linear (list[nnx.Module]): A list of feedforward layers including two linear transformations and dropout.
        norm1 (nnx.LayerNorm): Layer normalization after the multi-head attention layer.
        norm2 (nnx.LayerNorm): Layer normalization after the feedforward layers.
        dropout (nnx.Dropout): Dropout layer applied after the multi-head attention and feedforward layers.
    
    Args:
        input_dim (int): Dimensionality of the input embeddings.
        feedforward_dim (int): Dimensionality of the intermediate feedforward layer.
        dropout_prob (float): Probability of dropout.
        rngs (nnx.Rngs): Random number generators for reproducibility.
    """

    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int, 
                 dropout_prob: float, 
                 *, rngs: nnx.Rngs):
        self.mha = MultiHeadAttention(embed_dim=input_dim, rngs=rngs)
        self.linear = [
            nnx.Linear(input_dim, feedforward_dim, rngs=rngs),
            nnx.Dropout(dropout_prob, rngs=rngs),
            nnx.Linear(feedforward_dim, input_dim, rngs=rngs),
        ]
        self.norm1 = nnx.LayerNorm(input_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(input_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)

    def __call__(self, 
                 x: jnp.ndarray, 
                 num_heads: int = 8, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass for the Transformer encoder block.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, input_dim).
            num_heads (int): Number of attention heads in the multi-head attention mechanism. Default is 8.
            mask (Optional[jnp.ndarray]): Optional attention mask of shape 
                                          (seq_len, seq_len), 
                                          (batch_size, seq_len, seq_len), 
                                          or (batch_size, num_heads, seq_len, seq_len).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, seq_len, input_dim).
        """
        # Multi-Head Attention with residual connection and layer norm
        mha_out, _ = self.mha(x, num_heads=num_heads, mask=mask)
        x = x + self.dropout(mha_out)
        x = self.norm1(x)
        
        # Feedforward network with residual connection and layer norm
        linear_out = x
        for l in self.linear:
            linear_out = l(linear_out)
        linear_out = nnx.gelu(linear_out)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x



class TransformerEncoder(nnx.Module):
    """
    A Transformer encoder consisting of multiple stacked encoder blocks.

    Attributes:
        blocks (list[EncoderBlock]): List of encoder blocks comprising the Transformer encoder.

    Args:
        input_dim (int): Dimensionality of the input embeddings.
        feedforward_dim (int): Dimensionality of the intermediate feedforward layers in each encoder block.
        num_blocks (int): Number of encoder blocks to stack.
        dropout_prob (float): Probability of dropout.
        rngs (nnx.Rngs): Random number generators for reproducibility.
    """

    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int, 
                 num_blocks: int, 
                 dropout_prob: float, 
                 *, rngs: nnx.Rngs):    
        self.blocks = [
            EncoderBlock(input_dim, feedforward_dim, dropout_prob, rngs=rngs) 
            for _ in range(num_blocks)
        ]

    def __call__(self, 
                 x: jnp.ndarray, 
                 num_heads: int = 8, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass for the Transformer encoder.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, input_dim).
            num_heads (int): Number of attention heads for each block's multi-head attention. Default is 8.
            mask (Optional[jnp.ndarray]): Optional attention mask of shape 
                                          (seq_len, seq_len), 
                                          (batch_size, seq_len, seq_len), 
                                          or (batch_size, num_heads, seq_len, seq_len).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, seq_len, input_dim).
        """
        for block in self.blocks:
            x = block(x, num_heads=num_heads, mask=mask)
        return x

    def get_attention_weights(self, 
                              x: jnp.ndarray, 
                              num_heads: int = 8, 
                              mask: Optional[jnp.ndarray] = None
                              ) -> List[jnp.ndarray]:
        """
        Extracts the attention weights from each encoder block.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, input_dim).
            num_heads (int): Number of attention heads for each block's multi-head attention. Default is 8.
            mask (Optional[jnp.ndarray]): Optional attention mask of shape 
                                          (seq_len, seq_len), 
                                          (batch_size, seq_len, seq_len), 
                                          or (batch_size, num_heads, seq_len, seq_len).

        Returns:
            List[jnp.ndarray]: List of attention weight tensors from each encoder block. Each tensor has shape 
                               (batch_size, num_heads, seq_len, seq_len).
        """
        attention_weights = []
        for block in self.blocks:
            _, attention_weight = block.mha(x, num_heads=num_heads, mask=mask)
            attention_weights.append(attention_weight)
        return attention_weights


class PositionalEncoding(nnx.Module):
    """
    A module for computing positional encodings for input sequences.
    This is typically used in transformer models to provide information about
    the relative or absolute position of tokens in a sequence.

    Args:
        d_model (int): The dimension of the model (embedding size).
        max_seq_len (int): The maximum sequence length that the positional encodings
                           will support.
        rngs (nnx.Rngs): Random number generators for reproducibility.
    
    Attributes:
        pe (jnp.ndarray): The computed positional encodings of shape (1, max_seq_len, d_model).

    Methods:
        __call__(x: jnp.ndarray) -> jnp.ndarray:
            Adds the positional encodings to the input sequence.
    """
    
    def __init__(self, 
                 d_model: int, 
                 max_seq_len: int, 
                 *, rngs: nnx.Rngs):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model (embedding size).
            max_seq_len (int): The maximum sequence length for which positional
                               encodings are computed.
            rngs (nnx.Rngs): Random number generators for reproducibility.
        """
        # Initialize positional encoding array
        pe = jnp.zeros((max_seq_len, d_model))
        
        # Create position indices for the sequence
        position = jnp.arange(0, max_seq_len, dtype=jnp.float32)[:, jnp.newaxis]
        
        # Calculate the division term for the sine and cosine functions
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (jnp.log(10000) / d_model)) # exp(log(x)) = x !
        
        # Apply the sine and cosine functions to even and odd indices
        pe = pe.at[:, 0::2].set(jnp.sin(position / div_term))
        pe =pe.at[:, 1::2].set(jnp.cos(position / div_term))
        
        # Expand the positional encoding to have a batch dimension
        self.pe = jnp.expand_dims(pe, axis=0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Adds the positional encodings to the input sequence.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            jnp.ndarray: Output tensor with positional encodings added, of shape
                         (batch_size, seq_len, d_model).
        """
        # Add positional encodings to the input sequence
        return x + self.pe[:, :x.shape[1]]


class CrossMultiHeadAttention(nnx.Module):
    """
    Cross-attention mechanism using multi-head attention.
    
    This module computes attention between a query input (`x`) and key-value input (`kv`).
    It's often used in transformers for tasks like encoder-decoder attention.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        rngs (nnx.Rngs): Random number generators for reproducibility.
    
    Methods:
        __call__(x: jnp.ndarray, kv: jnp.ndarray, num_heads: Optional[int], mask: Optional[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Computes the cross-attention between `x` and `kv`.
    """
    def __init__(self, 
                 embed_dim: int, 
                 *, rngs: nnx.Rngs) -> None:
        """
        Initializes the CrossMultiHeadAttention module.

        Args:
            embed_dim (int): The dimensionality of input embeddings.
            rngs (nnx.Rngs): Random number generators for reproducibility.
        """
        self.kv_projection = nnx.Linear(embed_dim, 2 * embed_dim, rngs=rngs)
        self.q_projection = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.out_projection = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        
    def __call__(self, 
                 x: jnp.ndarray, 
                 kv: jnp.ndarray, 
                 num_heads: int, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Applies cross-attention between the query input (`x`) and key-value input (`kv`).

        Args:
            x (jnp.ndarray): Query tensor of shape (batch_size, seq_len, embed_dim).
            kv (jnp.ndarray): Key-value tensor of shape (batch_size, seq_len, embed_dim).
            num_heads (Optional[int]): Number of attention heads. Defaults to 1.
            mask (Optional[jnp.ndarray]): Optional attention mask of shape (batch_size, seq_len, seq_len).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: 
                - Attention output tensor of shape (batch_size, seq_len, embed_dim).
                - Attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        if mask is not None:
            mask = utils.expand_mask(mask)  # Ensure mask is in the correct 4D format
        
        # Project queries
        q = self.q_projection(x)
        
        # Project keys and values
        kv = self.kv_projection(kv)
        kv = kv.reshape(batch_size, -1, num_heads, 2*embed_dim // num_heads)
        kv = kv.transpose(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_len, d_k)
        k, v = jnp.array_split(kv, 2, axis=-1)  # Split into key and value
        
        # Split and reshape queries
        q = q.reshape(batch_size, seq_len, num_heads, -1)
        q = q.transpose(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_len, d_k)

        # Compute attention
        values, attention = scaled_dot_product(q, k, v, mask)
        
        # Reshape the output
        values = values.transpose(0, 2, 1, 3)  # Shape: (batch_size, seq_len, num_heads, d_k)
        values = values.reshape(batch_size, seq_len, embed_dim)  # Merge heads
        
        # Project output
        values = self.out_projection(values)
        
        return values, attention


class DecoderBlock(nnx.Module):
    """
    A single block for the transformer decoder.

    This module includes:
      - Self-attention.
      - Cross-attention between the decoder input and encoder output.
      - Feedforward layers with residual connections and normalization.

    Args:
        input_dim (int): The dimensionality of input embeddings.
        feedforward_dim (int): The dimensionality of the feedforward network.
        dropout_prob (float): Dropout probability for regularization.
        rngs (nnx.Rngs): Random number generators for reproducibility.

    Methods:
        __call__(x, encoder_kv, num_heads, mask) -> jnp.ndarray:
            Performs forward computation of the decoder block.
    """
    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int,
                 dropout_prob: float,
                 *, rngs: nnx.Rngs) -> None:
        """
        Initializes the DecoderBlock module.
        """
        self.mha = MultiHeadAttention(embed_dim=input_dim, rngs=rngs)  # Self-attention
        self.cmha = CrossMultiHeadAttention(embed_dim=input_dim, rngs=rngs)  # Cross-attention
        self.linear = [
            nnx.Linear(input_dim, feedforward_dim, rngs=rngs),
            nnx.Dropout(dropout_prob, rngs=rngs),
            nnx.Linear(feedforward_dim, input_dim, rngs=rngs),
        ]
        self.norm1 = nnx.LayerNorm(input_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(input_dim, rngs=rngs)
        self.norm3 = nnx.LayerNorm(input_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)
        
    def __call__(self, 
                 x: jnp.ndarray, 
                 encoder_kv: jnp.ndarray, 
                 num_heads: int, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass for the DecoderBlock.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, input_dim).
            encoder_kv (jnp.ndarray): Encoder output used as key-value pairs for cross-attention.
            num_heads (int): Number of attention heads.
            mask (Optional[jnp.ndarray]): Optional mask for attention.

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, seq_len, input_dim).
        """
        # Self-attention
        mha_out, _ = self.mha(x, num_heads=num_heads, mask=mask)
        x = x + self.dropout(mha_out)
        x = self.norm1(x)
        
        # Cross-attention
        #cmha_out, _ = self.cmha(x, encoder_kv, num_heads=num_heads, mask=mask)
        cmha_out, _ = self.cmha(x, encoder_kv, num_heads=num_heads)
        x = x + self.dropout(cmha_out)
        x = self.norm2(x)
        
        # Feedforward
        linear_out = x
        for l in self.linear:
            linear_out = l(linear_out)
        linear_out = nnx.gelu(linear_out)
        x = x + self.dropout(linear_out)
        x = self.norm3(x)
        return x
    


class TransformerDecoder(nnx.Module):
    """
    Transformer decoder module consisting of multiple decoder blocks and an output projection layer.

    Args:
        input_dim (int): Dimensionality of the input embeddings.
        feedforward_dim (int): Dimensionality of the feedforward network.
        num_blocks (int): Number of decoder blocks.
        dropout_prob (float): Dropout probability for regularization.
        rngs (nnx.Rngs): Random number generators for reproducibility.

    Methods:
        __call__(x, encoder_kv, num_heads, mask) -> jnp.ndarray:
            Performs the forward computation through the decoder.

        get_mha_attention_weights(x, num_heads, mask) -> List[jnp.ndarray]:
            Returns a list of self-attention weight matrices from all decoder blocks.

        get_cmha_attention_weights(x, encoder_kv, num_heads, mask) -> List[jnp.ndarray]:
            Returns a list of cross-attention weight matrices from all decoder blocks.
    """
    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int,
                 num_blocks: int,
                 dropout_prob: float,
                 *, rngs: nnx.Rngs) -> None:
        """
        Initializes the TransformerDecoder module.
        """
        self.blocks = [
            DecoderBlock(input_dim, feedforward_dim, dropout_prob, rngs=rngs)
            for _ in range(num_blocks)
        ]
        self.out_projection = nnx.Linear(input_dim, input_dim, rngs=rngs)
        
    def __call__(self, 
                 x: jnp.ndarray, 
                 encoder_kv: jnp.ndarray, 
                 num_heads: int, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass for the TransformerDecoder.

        Args:
            x (jnp.ndarray): Decoder input tensor of shape (batch_size, seq_len, input_dim).
            encoder_kv (jnp.ndarray): Encoder output used as key-value pairs for cross-attention.
            num_heads (int): Number of attention heads.
            mask (Optional[jnp.ndarray]): Optional mask for attention.

        Returns:
            jnp.ndarray: Output tensor after applying the decoder.
        """
        for block in self.blocks:
            x = block(x, encoder_kv, num_heads, mask)
        #x = self.out_projection(x)
        #x = nnx.softmax(x, axis=-1)  # Optional final softmax
        return x
    
    def get_mha_attention_weights(self, 
                                  x: jnp.ndarray, 
                                  num_heads: int = 8, 
                                  mask: Optional[jnp.ndarray] = None
                                  ) -> List[jnp.ndarray]:
        """
        Collects self-attention weights from all decoder blocks.

        Args:
            x (jnp.ndarray): Decoder input tensor.
            num_heads (int): Number of attention heads.
            mask (Optional[jnp.ndarray]): Optional attention mask.

        Returns:
            List[jnp.ndarray]: List of self-attention weights from all blocks.
        """
        attention_weights = []
        for block in self.blocks:
            _, attention_weight = block.mha(x, num_heads=num_heads, mask=mask)
            attention_weights.append(attention_weight)
        return attention_weights
    
    def get_cmha_attention_weights(self, 
                                   x: jnp.ndarray, 
                                   encoder_kv: jnp.ndarray,
                                   num_heads: int = 8, 
                                   mask: Optional[jnp.ndarray] = None
                                   ) -> List[jnp.ndarray]:
        """
        Collects cross-attention weights from all decoder blocks.

        Args:
            x (jnp.ndarray): Decoder input tensor.
            encoder_kv (jnp.ndarray): Encoder output used for cross-attention.
            num_heads (int): Number of attention heads.
            mask (Optional[jnp.ndarray]): Optional attention mask.

        Returns:
            List[jnp.ndarray]: List of cross-attention weights from all blocks.
        """
        attention_weights = []
        for block in self.blocks:
            _, attention_weight = block.cmha(x, encoder_kv, num_heads=num_heads, mask=mask)
            attention_weights.append(attention_weight)
        return attention_weights


class Transformer(nnx.Module):
    """
    Transformer module combining the encoder and decoder.

    Args:
        input_dim (int): Dimensionality of the input embeddings.
        feedforward_dim (int): Dimensionality of the feedforward network.
        num_blocks (int): Number of encoder and decoder blocks.
        dropout_prob (float): Dropout probability for regularization.
        rngs (nnx.Rngs): Random number generators for reproducibility.

    Methods:
        __call__(x, y, num_heads, mask=None) -> jnp.ndarray:
            Performs the forward computation through the Transformer.
    """
    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int, 
                 num_blocks: int, 
                 dropout_prob: float, 
                 *, rngs: nnx.Rngs) -> None:
        """
        Initializes the Transformer module.
        """
        self.encoder = TransformerEncoder(
            input_dim=input_dim, 
            feedforward_dim=feedforward_dim, 
            num_blocks=num_blocks, 
            dropout_prob=dropout_prob, 
            rngs=rngs
        )
        self.decoder = TransformerDecoder(
            input_dim=input_dim, 
            feedforward_dim=feedforward_dim, 
            num_blocks=num_blocks, 
            dropout_prob=dropout_prob, 
            rngs=rngs
        )
        self.out_projection = nnx.Linear(input_dim, input_dim, rngs=rngs)

    def __call__(self, 
                 x: jnp.ndarray, 
                 y: jnp.ndarray, 
                 num_heads: int, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass for the Transformer.

        Args:
            x (jnp.ndarray): Input tensor (source sequence) of shape (batch_size, src_seq_len, input_dim).
            y (jnp.ndarray): Target tensor (shifted target sequence) of shape (batch_size, tgt_seq_len, input_dim).
            num_heads (int): Number of attention heads.
            mask (Optional[jnp.ndarray]): Optional mask for attention.

        Returns:
            jnp.ndarray: Output tensor after applying the Transformer.
        """
        # Pass through encoder
        encoder_kv = self.encoder(x, num_heads)
        
        # Pass through decoder with encoder outputs
        decoder_output = self.decoder(y, encoder_kv, num_heads, mask)
        
        # Final linear projection
        output = self.out_projection(decoder_output)
        return output
