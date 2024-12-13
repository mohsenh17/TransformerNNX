import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from flax import nnx

from model import utils

def compute_positional_encoding(d_model: int, max_seq_len: int = 512) -> jnp.ndarray:
    """
    Computes the positional encoding for a given sequence length and embedding dimension.
    
    Args:
        d_model (int): The dimension of the model (embedding size).
        max_seq_len (int): The maximum sequence length for which positional encodings
                           are computed.
    
    Returns:
        jnp.ndarray: The computed positional encodings of shape (1, max_seq_len, d_model).
    """
    # Initialize positional encoding array
    pe = jnp.zeros((max_seq_len, d_model))
    
    # Create position indices for the sequence
    position = jnp.arange(0, max_seq_len, dtype=jnp.float32)[:, jnp.newaxis]
    
    # Calculate the division term for the sine and cosine functions
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (jnp.log(10000.0) / d_model))  # exp(log(x)) = x
    
    # Apply the sine and cosine functions to even and odd indices
    pe = pe.at[:, 0::2].set(jnp.sin(position / div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position / div_term))
    
    # Expand the positional encoding to have a batch dimension
    pe = jnp.expand_dims(pe, axis=0)
    
    return pe

import jax.numpy as jnp

def compute_relative_positional_encoding(max_seq_len: int) -> jnp.ndarray:
    """
    Computes the relative positional encoding for a sequence of given length.

    Args:
        max_seq_len (int): The maximum sequence length.

    Returns:
        jnp.ndarray: A 2D array of shape (max_seq_len, max_seq_len) representing
                     the relative positional encoding.
    """
    # Create the positional indices
    pe = jnp.arange(max_seq_len)
    
    # Compute the relative positional encoding (RPE)
    rpe = pe - pe[:, jnp.newaxis]  # Shape: (max_seq_len, max_seq_len)
    
    # Offset the RPE to ensure non-negative values
    rpe += max_seq_len
    
    return rpe

def scaled_dot_product(
    q: jnp.ndarray, 
    k: jnp.ndarray, 
    v: jnp.ndarray, 
    mask: Optional[jnp.ndarray] = None,
    rpem: Optional[jnp.ndarray] = None
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

    if rpem is not None:
        bias = jnp.einsum("bhsd,hskd->bhsk", q, rpem) 
        attn_logits = attn_logits + bias

    if mask is not None:
        attn_logits = jnp.where(mask == 0, -1e9, attn_logits)  # Apply mask with large negative value

    attention = jax.nn.softmax(attn_logits, axis=-1)  # Softmax over last axis
    values = jnp.matmul(attention, v)  # Compute weighted values
    return values, attention

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
                 max_seq_len: int=512, 
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
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
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
        pe = compute_positional_encoding(self.d_model, self.max_seq_len)
        return x + pe[:, :x.shape[1]]

class XLRelativePositionalEncoding(nnx.Module):
    """
    A module for computing relative positional encodings for input sequences.
    This is typically used in transformer models to provide information about
    the relative position of tokens in a sequence.

    Relative positional encoding helps a model identify the relative distance
    between tokens, enabling the attention mechanism to understand sequence
    order without absolute positional embeddings.

    Attributes:
        embedding (nnx.Embed): A trainable embedding layer that maps relative
                               positions to embeddings of shape 
                               (2 * max_seq_len - 1, d_model).

    Args:
        d_model (int): The dimensionality of the model (embedding size).
        max_seq_len (int): The maximum sequence length that the positional encodings
                           will support.
        rngs (nnx.Rngs): Random number generators for initializing the embedding weights.

    Methods:
        __call__(): Returns the relative positional embeddings for all token pairs.
    """
    def __init__(self, 
                 d_model: int, 
                 max_seq_len: int=512, 
                 *, rngs: nnx.Rngs):
        """
        Initializes the relative positional encoding module.

        Args:
            d_model (int): The dimensionality of the model (embedding size).
            max_seq_len (int): The maximum sequence length that the positional encodings
                               will support.
            rngs (nnx.Rngs): Random number generators for reproducibility.
        """
        self.d_model = d_model
        # Trainable embedding layer for relative positions
        self.embedding = nnx.Embed(2*max_seq_len-1, d_model, rngs=rngs) # shape: (2*max_seq_len-1, d_model)


    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the relative positional encodings for a sequence.

        Returns:
            jnp.ndarray: A tensor of shape (max_seq_len, max_seq_len, d_model)
                         containing relative positional embeddings for all token pairs.
        """
        seq_len = x.shape[1]
        rpe = compute_relative_positional_encoding(seq_len)
        return self.embedding(rpe) # shape: (max_seq_len, max_seq_len, d_model)

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

    def __init__(self, 
                 embed_dim: int, 
                 num_heads:int, 
                 relative_positional_encoding_flag: bool = False,
                 *, rngs: nnx.Rngs) -> None:
        """
        Initializes the MultiHeadAttention module.

        Parameters:
        ----------
        embed_dim : int
            The dimension of the input embeddings.
        num_heads : int
            The number of attention heads.
        rngs : nnx.Rngs
            Random number generators for initializing weights of the projection layers.
        """
        self.num_heads = num_heads
        #self.rngs = rngs
        self.rpef = relative_positional_encoding_flag
        self.qkv_projection = nnx.Linear(embed_dim, 3 * embed_dim, rngs=rngs)
        self.out_projection = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        if self.rpef:
            self.rpem = XLRelativePositionalEncoding(embed_dim, rngs=rngs)
        
        

    def __call__(self, 
                 x: jnp.ndarray, 
                 mask: Optional[jnp.ndarray] = None,
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Applies the multi-head attention mechanism.

        Parameters:
        ----------
        x : jnp.ndarray
            Input tensor of shape `(batch_size, seq_len, embed_dim)`.
        mask : Optional[jnp.ndarray], default=None
            Optional mask tensor of shape `(batch_size, seq_len, seq_len)` 
            or `(batch_size, num_heads, seq_len, seq_len)` to apply attention masking.

        Returns:
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            - Output tensor of shape `(batch_size, seq_len, embed_dim)`.
            - Attention weights tensor of shape `(batch_size, num_heads, seq_len, seq_len)`.
        """
        rpem = None
        batch_size, seq_len, embed_dim = x.shape
        if mask is not None:
            mask = utils.expand_mask(mask)  # Ensure mask is in the correct 4D format
        if self.rpef:
            rpem = self.rpem(x).reshape(self.num_heads, seq_len, seq_len, -1)
        # Compute query, key, and value projections
        qkv = self.qkv_projection(x)  # Shape: (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, -1)  # Split heads
        qkv = qkv.transpose(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_len, d_k)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)  # Split into query, key, value
        
        # Compute scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask, rpem)  # Custom function
        
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
        num_heads (int): Number of attention heads.
        rngs (nnx.Rngs): Random number generators for reproducibility.
    """

    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int, 
                 dropout_prob: float, 
                 num_heads: int,
                 relative_positional_encoding_flag: bool = False,
                 *, rngs: nnx.Rngs):
        self.mha = MultiHeadAttention(input_dim, num_heads,relative_positional_encoding_flag, rngs=rngs)
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
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass for the Transformer encoder block.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, input_dim).
            mask (Optional[jnp.ndarray]): Optional attention mask of shape 
                                          (seq_len, seq_len), 
                                          (batch_size, seq_len, seq_len), 
                                          or (batch_size, num_heads, seq_len, seq_len).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, seq_len, input_dim).
        """
        # Multi-Head Attention with residual connection and layer norm
        mha_out, _ = self.mha(x, mask=mask)
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
        num_heads (int): Number of attention heads in each encoder block.
        rngs (nnx.Rngs): Random number generators for reproducibility.
    """

    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int, 
                 num_blocks: int, 
                 dropout_prob: float, 
                 num_heads: int,
                 relative_positional_encoding_flag: bool = False,
                 *, rngs: nnx.Rngs):    
        self.blocks = [
            EncoderBlock(input_dim, feedforward_dim, dropout_prob, num_heads, relative_positional_encoding_flag, rngs=rngs) 
            for _ in range(num_blocks)
        ]

    def __call__(self, 
                 x: jnp.ndarray, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass for the Transformer encoder.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, input_dim).
            mask (Optional[jnp.ndarray]): Optional attention mask of shape 
                                          (seq_len, seq_len), 
                                          (batch_size, seq_len, seq_len), 
                                          or (batch_size, num_heads, seq_len, seq_len).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, seq_len, input_dim).
        """
        for block in self.blocks:
            x = block(x, mask=mask)
        return x

    def get_attention_weights(self, 
                              x: jnp.ndarray, 
                              mask: Optional[jnp.ndarray] = None,
                              ) -> List[jnp.ndarray]:
        """
        Extracts the attention weights from each encoder block.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, input_dim).
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
            _, attention_weight = block.mha(x, mask=mask)
            attention_weights.append(attention_weight)
        return attention_weights



class CrossMultiHeadAttention(nnx.Module):
    """
    Cross-attention mechanism using multi-head attention.
    
    This module computes attention between a query input (`x`) and key-value input (`kv`).
    It's often used in transformers for tasks like encoder-decoder attention.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        num_heads (int): Number of attention heads.
        rngs (nnx.Rngs): Random number generators for reproducibility.
    
    Methods:
        __call__(x: jnp.ndarray, kv: jnp.ndarray, num_heads: Optional[int], mask: Optional[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Computes the cross-attention between `x` and `kv`.
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int,
                 *, rngs: nnx.Rngs) -> None:
        """
        Initializes the CrossMultiHeadAttention module.

        Args:
            embed_dim (int): The dimensionality of input embeddings.
            rngs (nnx.Rngs): Random number generators for reproducibility.
        """
        self.num_heads = num_heads
        self.kv_projection = nnx.Linear(embed_dim, 2 * embed_dim, rngs=rngs)
        self.q_projection = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.out_projection = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        
    def __call__(self, 
                 x: jnp.ndarray, 
                 kv: jnp.ndarray, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Applies cross-attention between the query input (`x`) and key-value input (`kv`).

        Args:
            x (jnp.ndarray): Query tensor of shape (batch_size, seq_len, embed_dim).
            kv (jnp.ndarray): Key-value tensor of shape (batch_size, seq_len, embed_dim).
            mask (Optional[jnp.ndarray]): Optional attention mask of shape (batch_size, seq_len, seq_len).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: 
                - Attention output tensor of shape (batch_size, seq_len, embed_dim).
                - Attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        if mask is not None:
            mask = utils.expand_mask(mask)  # Ensure mask is in the correct 4D format
        
        # Project queries
        q = self.q_projection(x)
        
        # Project keys and values
        kv = self.kv_projection(kv)
        kv = kv.reshape(batch_size, -1, self.num_heads, 2*embed_dim // self.num_heads)
        kv = kv.transpose(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_len, d_k)
        k, v = jnp.array_split(kv, 2, axis=-1)  # Split into key and value
        
        # Split and reshape queries
        q = q.reshape(batch_size, seq_len, self.num_heads, -1)
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
        num_heads (int): Number of attention heads.
        rngs (nnx.Rngs): Random number generators for reproducibility.

    Methods:
        __call__(x, encoder_kv, mask) -> jnp.ndarray:
            Performs forward computation of the decoder block.
    """
    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int,
                 dropout_prob: float,
                 num_heads: int,
                 relative_positional_encoding_flag: bool = False,
                 *, rngs: nnx.Rngs) -> None:
        """
        Initializes the DecoderBlock module.
        """
        self.mha = MultiHeadAttention(input_dim,num_heads,relative_positional_encoding_flag, rngs=rngs)  # Self-attention
        self.cmha = CrossMultiHeadAttention(input_dim, num_heads, rngs=rngs)  # Cross-attention
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
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass for the DecoderBlock.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, seq_len, input_dim).
            encoder_kv (jnp.ndarray): Encoder output used as key-value pairs for cross-attention.
            mask (Optional[jnp.ndarray]): Optional mask for attention.

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, seq_len, input_dim).
        """
        # Self-attention
        mha_out, _ = self.mha(x, mask=mask)
        x = x + self.dropout(mha_out)
        x = self.norm1(x)
        
        # Cross-attention
        #cmha_out, _ = self.cmha(x, encoder_kv, mask=mask)
        cmha_out, _ = self.cmha(x, encoder_kv)
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
        num_heads (int): Number of attention heads.
        rngs (nnx.Rngs): Random number generators for reproducibility.

    Methods:
        __call__(x, encoder_kv, mask) -> jnp.ndarray:
            Performs the forward computation through the decoder.

        get_mha_attention_weights(x, mask) -> List[jnp.ndarray]:
            Returns a list of self-attention weight matrices from all decoder blocks.

        get_cmha_attention_weights(x, encoder_kv, mask) -> List[jnp.ndarray]:
            Returns a list of cross-attention weight matrices from all decoder blocks.
    """
    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int,
                 num_blocks: int,
                 dropout_prob: float,
                 num_heads: int,
                 relative_positional_encoding_flag: bool = False,
                 *, rngs: nnx.Rngs) -> None:
        """
        Initializes the TransformerDecoder module.
        """
        self.blocks = [
            DecoderBlock(input_dim, feedforward_dim, dropout_prob, num_heads,relative_positional_encoding_flag, rngs=rngs)
            for _ in range(num_blocks)
        ]
        self.out_projection = nnx.Linear(input_dim, input_dim, rngs=rngs)
        
    def __call__(self, 
                 x: jnp.ndarray, 
                 encoder_kv: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass for the TransformerDecoder.

        Args:
            x (jnp.ndarray): Decoder input tensor of shape (batch_size, seq_len, input_dim).
            encoder_kv (jnp.ndarray): Encoder output used as key-value pairs for cross-attention.
            mask (Optional[jnp.ndarray]): Optional mask for attention.

        Returns:
            jnp.ndarray: Output tensor after applying the decoder.
        """
        for block in self.blocks:
            x = block(x, encoder_kv, mask)
        #x = self.out_projection(x)
        #x = nnx.softmax(x, axis=-1)  # Optional final softmax
        return x
    
    def get_mha_attention_weights(self, 
                                  x: jnp.ndarray, 
                                  mask: Optional[jnp.ndarray] = None
                                  ) -> List[jnp.ndarray]:
        """
        Collects self-attention weights from all decoder blocks.

        Args:
            x (jnp.ndarray): Decoder input tensor.
            mask (Optional[jnp.ndarray]): Optional attention mask.

        Returns:
            List[jnp.ndarray]: List of self-attention weights from all blocks.
        """
        attention_weights = []
        for block in self.blocks:
            _, attention_weight = block.mha(x, mask=mask)
            attention_weights.append(attention_weight)
        return attention_weights
    
    def get_cmha_attention_weights(self, 
                                   x: jnp.ndarray, 
                                   encoder_kv: jnp.ndarray,
                                   mask: Optional[jnp.ndarray] = None
                                   ) -> List[jnp.ndarray]:
        """
        Collects cross-attention weights from all decoder blocks.

        Args:
            x (jnp.ndarray): Decoder input tensor.
            encoder_kv (jnp.ndarray): Encoder output used for cross-attention.
            mask (Optional[jnp.ndarray]): Optional attention mask.

        Returns:
            List[jnp.ndarray]: List of cross-attention weights from all blocks.
        """
        attention_weights = []
        for block in self.blocks:
            _, attention_weight = block.cmha(x, encoder_kv, mask=mask)
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
        num_heads (int): Number of attention heads.
        rngs (nnx.Rngs): Random number generators for reproducibility.

    Methods:
        __call__(x, y, mask=None) -> jnp.ndarray:
            Performs the forward computation through the Transformer.
    """
    def __init__(self, 
                 input_dim: int, 
                 feedforward_dim: int, 
                 num_blocks: int, 
                 dropout_prob: float, 
                 num_heads: int,
                 *, rngs: nnx.Rngs) -> None:
        """
        Initializes the Transformer module.
        """
        self.encoder = TransformerEncoder(
            input_dim=input_dim, 
            feedforward_dim=feedforward_dim, 
            num_blocks=num_blocks, 
            dropout_prob=dropout_prob, 
            num_heads=num_heads,
            rngs=rngs
        )
        self.decoder = TransformerDecoder(
            input_dim=input_dim, 
            feedforward_dim=feedforward_dim, 
            num_blocks=num_blocks, 
            dropout_prob=dropout_prob, 
            num_heads=num_heads,
            rngs=rngs
        )
        self.out_projection = nnx.Linear(input_dim, input_dim, rngs=rngs)

    def __call__(self, 
                 x: jnp.ndarray, 
                 y: jnp.ndarray, 
                 mask: Optional[jnp.ndarray] = None
                 ) -> jnp.ndarray:
        """
        Forward pass for the Transformer.

        Args:
            x (jnp.ndarray): Input tensor (source sequence) of shape (batch_size, src_seq_len, input_dim).
            y (jnp.ndarray): Target tensor (shifted target sequence) of shape (batch_size, tgt_seq_len, input_dim).
            mask (Optional[jnp.ndarray]): Optional mask for attention.

        Returns:
            jnp.ndarray: Output tensor after applying the Transformer.
        """
        # Pass through encoder
        encoder_kv = self.encoder(x)
        
        # Pass through decoder with encoder outputs
        decoder_output = self.decoder(y, encoder_kv, mask)
        
        # Final linear projection
        output = self.out_projection(decoder_output)
        return output

if __name__ == '__main__':
    pe = XLRelativePositionalEncoding(8, 4, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 3, 8))
    print(pe(x).shape)