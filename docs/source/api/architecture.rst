architecture module
===================

# Transformer Documentation

The Transformer model is a powerful and flexible neural network architecture designed to process sequential data. It is widely used in natural language processing (NLP), computer vision, and other domains requiring attention-based modeling. This document provides a detailed overview of various components and modules within the Transformer architecture.

---

## Positional Encoding

### Absolute Positional Encoding

Absolute positional encoding provides information about the position of tokens within a sequence. This is essential for models like transformers, which lack inherent sequential processing.

#### Function: `compute_positional_encoding`

**Description:**
Computes the absolute positional encoding for a given sequence length and embedding dimension.

**Parameters:**
- `d_model` (int): The dimension of the model (embedding size).
- `max_seq_len` (int): The maximum sequence length for which positional encodings are computed.

**Returns:**
- `jnp.ndarray`: A tensor of shape `(1, max_seq_len, d_model)` containing positional encodings.

#### Class: `PositionalEncoding`

**Description:**
A module for computing and adding positional encodings to input sequences. This module is commonly used in transformer models.

**Attributes:**
- `pe` (`jnp.ndarray`): Precomputed positional encodings of shape `(1, max_seq_len, d_model)`.

**Methods:**
- `__call__(x: jnp.ndarray) -> jnp.ndarray`: Adds positional encodings to the input tensor.

---

### Relative Positional Encoding

Relative positional encoding allows the model to understand the relative distance between tokens, enhancing its ability to process long sequences effectively.

#### Function: `compute_relative_positional_encoding`

**Description:**
Computes the relative positional encoding for a sequence of given length.

**Parameters:**
- `max_seq_len` (int): The maximum sequence length.

**Returns:**
- `jnp.ndarray`: A 2D array of shape `(max_seq_len, max_seq_len)` representing relative positional encoding.

#### Class: `RelativePositionalEncoding`

**Description:**
A module for computing relative positional encodings, used for capturing relationships between tokens in a sequence.

**Attributes:**
- `embedding` (`nnx.Embed`): A trainable embedding layer for mapping relative positions to embeddings.

**Methods:**
- `__call__() -> jnp.ndarray`: Returns relative positional encodings for all token pairs.

---

## Multi-Head Attention

Multi-head attention is a key component of the transformer architecture, enabling the model to focus on different parts of the input sequence.

#### Function: `compute_scaled_dot_product_attention`

**Description:**
Computes scaled dot-product attention.

**Parameters:**
- `q` (tensor): Query tensor of shape `(..., seq_len, d_k)`.
- `k` (tensor): Key tensor of shape `(..., seq_len, d_k)`.
- `v` (tensor): Value tensor of shape `(..., seq_len, d_k)`.
- `mask` (optional, tensor): Mask tensor of shape `(..., seq_len, seq_len)`.

**Returns:**
- `values` (tensor): Tensor of shape `(..., seq_len, d_k)` containing attention-weighted values.
- `attention` (tensor): Tensor of shape `(..., seq_len, seq_len)` containing attention scores.

#### Class: `MultiHeadAttention`

**Description:**
Implements a multi-head attention mechanism with trainable projection layers.

**Attributes:**
- `qkv_projection` (`nnx.Linear`): Projects the input into query, key, and value tensors.
- `out_projection` (`nnx.Linear`): Projects the output back to the embedding dimension.

**Methods:**
- `__call__(x, mask=None) -> Tuple[jnp.ndarray, jnp.ndarray]`: Computes attention outputs and attention scores.

---

## Transformer Encoder

The encoder processes input sequences to produce contextualized embeddings. It consists of stacked encoder blocks, each containing multi-head attention, feedforward layers, and normalization.

#### Class: `EncoderBlock`

**Description:**
A single encoder block with self-attention, feedforward layers, and dropout.

**Attributes:**
- `mha` (`MultiHeadAttention`): Multi-head attention mechanism.
- `linear` (list): Feedforward layers.
- `norm1` (`nnx.LayerNorm`): Normalization after attention.
- `norm2` (`nnx.LayerNorm`): Normalization after feedforward layers.
- `dropout` (`nnx.Dropout`): Dropout regularization.

**Methods:**
- `__call__(x, mask=None) -> jnp.ndarray`: Processes input tensors with attention and feedforward layers.

#### Class: `TransformerEncoder`

**Description:**
A transformer encoder consisting of multiple stacked encoder blocks.

**Attributes:**
- `blocks` (list): List of encoder blocks.

**Methods:**
- `__call__(x, mask=None) -> jnp.ndarray`: Processes input tensors through the encoder.
- `extract_attention_weights(x, mask=None) -> List[jnp.ndarray]`: Extracts attention weights from encoder blocks.

---

## Transformer Decoder

The decoder processes target sequences with attention to both the target sequence and encoder outputs.

#### Class: `DecoderBlock`

**Description:**
A single decoder block with self-attention, cross-attention, feedforward layers, and normalization.

**Methods:**
- `__call__(x, encoder_kv, mask=None) -> jnp.ndarray`: Processes input with attention mechanisms and feedforward layers.

#### Class: `TransformerDecoder`

**Description:**
A transformer decoder with multiple decoder blocks and an output projection layer.

**Methods:**
- `__call__(x, encoder_kv, mask=None) -> jnp.ndarray`: Processes input sequences through the decoder.
- `get_self_attention_weights(x, mask=None) -> List[jnp.ndarray]`: Extracts self-attention weights from decoder blocks.
- `get_cross_attention_weights(x, encoder_kv, mask=None) -> List[jnp.ndarray]`: Extracts cross-attention weights from decoder blocks.

---

## Transformer Model

#### Class: `Transformer`

**Description:**
The complete transformer model, combining the encoder and decoder.

**Attributes:**
- `encoder` (`TransformerEncoder`): The encoder component.
- `decoder` (`TransformerDecoder`): The decoder component.

**Methods:**
- `__call__(x, y, mask=None) -> jnp.ndarray`: Processes input and target sequences through the transformer.



.. automodule:: architecture
   :members:
   :undoc-members:
   :show-inheritance:
