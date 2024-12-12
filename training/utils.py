from typing import Tuple, Dict, Optional, List
from flax import nnx
import optax
import jax.numpy as jnp
import jax
import numpy as np


from typing import Dict
import jax.numpy as jnp

def generate_sequence(model:nnx.Module, batch: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a sequence using an encoder-decoder model.

    Args:
        model: The model containing the embedding projection, encoder, decoder, and output projection.
        batch (Dict[str, jnp.ndarray]): A dictionary containing batch data, including 'targets' of shape 
                                        (batch_size, seq_length, input_dim).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - Predicted sequence of shape (batch_size, seq_length), where each entry is the predicted token ID.
            - Final decoder output tensor of shape (batch_size, seq_length, input_dim), containing logits or embeddings.
    """
    batch_size, seq_length, input_dim = batch['targets'].shape

    # Initialize predictions array and the initial decoder input (start token embedding)
    preds = jnp.zeros((batch_size, seq_length), dtype=jnp.int32)
    y = jnp.zeros((batch_size, 1, input_dim))  # Start token for decoder input

    # Encoder output (fixed for the entire sequence generation)
    embd_output = model.embd_projection(batch['targets'])
    encoder_output = model.model_backbone.encoder(embd_output)

    for t in range(seq_length):
        # Decoder processes the current sequence
        embd_target_output = model.embd_projection(y)
        decoder_output = model.model_backbone.decoder(embd_target_output, encoder_output)
        output = model.output_projection(decoder_output)

        # Predict the next token (argmax over vocabulary dimension)
        next_token = output[:, -1, :].argmax(axis=-1)

        # Store the predicted token
        preds = preds.at[:, t].set(next_token)

        # Update decoder input for the next time step
        next_token_embedding = jnp.eye(input_dim)[next_token]  # One-hot embedding
        y = jnp.concatenate([y, next_token_embedding[:, None, :]], axis=1)

    return preds, output



def loss_fn(model: nnx.Module, 
            encoder_inputs: jnp.ndarray, 
            decoder_inputs: jnp.ndarray, 
            targets: jnp.ndarray, 
            mask: Optional[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the loss and logits for a given model and inputs.

    Args:
        model (nnx.Module): The Flax model.
        encoder_inputs (jnp.ndarray): One-hot encoded encoder inputs of shape (batch_size, seq_length, vocab_size).
        decoder_inputs (jnp.ndarray): One-hot encoded decoder inputs of shape (batch_size, seq_length, vocab_size).
        targets (jnp.ndarray): One-hot encoded target labels of shape (batch_size, seq_length, vocab_size).
        mask (Optional[jnp.ndarray]): Optional attention mask of shape (batch_size, seq_length, seq_length).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - loss: Scalar loss value.
            - logits: Model predictions of shape (batch_size, seq_length, vocab_size).
    """
    logits = model(encoder_inputs, decoder_inputs, mask)
    loss = optax.softmax_cross_entropy(logits=logits, labels=targets).mean()
    return loss, logits

def transformer_loss_fn(logits: jnp.ndarray, 
            targets: jnp.ndarray, 
            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the loss and logits for a given model and inputs.

    Args:
        model (nnx.Module): The Flax model.
        encoder_inputs (jnp.ndarray): One-hot encoded encoder inputs of shape (batch_size, seq_length, vocab_size).
        decoder_inputs (jnp.ndarray): One-hot encoded decoder inputs of shape (batch_size, seq_length, vocab_size).
        targets (jnp.ndarray): One-hot encoded target labels of shape (batch_size, seq_length, vocab_size).
        mask (Optional[jnp.ndarray]): Optional attention mask of shape (batch_size, seq_length, seq_length).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - loss: Scalar loss value.
            - logits: Model predictions of shape (batch_size, seq_length, vocab_size).
    """
    #logits = model(encoder_inputs, decoder_inputs, mask)
    loss = optax.softmax_cross_entropy(logits=logits, labels=targets).mean()
    return loss, logits


@nnx.jit
def train_step(model: nnx.Module, 
               optimizer: nnx.Optimizer, 
               metrics: Dict, 
               batch: Dict[str, jnp.ndarray], 
               mask: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Performs a single training step, updating model parameters.

    Args:
        model (nnx.Module): The Flax model to be trained.
        optimizer (nnx.Optimizer): The optimizer for parameter updates.
        metrics (nnx.Metrics): Metrics object to update with training statistics.
        batch (Dict[str, jnp.ndarray]): A batch of input data containing:
            - 'encoder_inputs': Input data for the encoder.
            - 'decoder_inputs': Input data for the decoder.
            - 'targets': Target labels.
        mask (Optional[jnp.ndarray]): Optional attention mask.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - loss: Scalar loss value for the batch.
            - logits: Model predictions after applying the sigmoid activation.
    """
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(
        model, 
        batch['encoder_inputs'], 
        batch['decoder_inputs'], 
        batch['targets'], 
        mask=mask
    )
    metrics.update(loss=loss, logits=logits, labels=batch['targets'])
    optimizer.update(grads)
    return loss, nnx.softmax(logits)

@nnx.jit
def eval_step(model: nnx.Module, 
              metrics: Dict, 
              batch: Dict[str, jnp.ndarray],
              mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Performs a single evaluation step for a given model and batch of data.

    Args:
        model (nnx.Module): The model instance, which can include different architectures.
        metrics (Dict[str, jnp.ndarray]): A dictionary to store metrics such as loss and logits.
        batch (Dict[str, jnp.ndarray]): Input batch containing:
            - 'encoder_inputs': Inputs for the encoder.
            - 'decoder_inputs': Inputs for the decoder.
            - 'targets': Ground truth labels.
        mask (Optional[jnp.ndarray]): Optional attention mask for sequences.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - Computed loss for the batch.
            - Softmax probabilities of the model's predictions.
    """
    if model.model_backbone.__class__.__name__ == 'Transformer':
        preds, output = generate_sequence(model, batch)

        loss, logits = transformer_loss_fn(
            output,
            batch['targets'], 
        )
    else:
        loss, logits = loss_fn(
        model, 
        batch['encoder_inputs'], 
        batch['decoder_inputs'], 
        batch['targets'], 
        mask=mask
    )
    

    metrics.update(loss=loss, logits=logits, labels=batch['targets'])
    return loss, nnx.softmax(logits)

    

@nnx.jit(static_argnums=(2,))
def pred_step(model: nnx.Module, 
              batch: Dict[str, jnp.ndarray], 
              max_seq_len: int,
              ) -> jnp.ndarray:
    """
    Performs autoregressive prediction using the given transformer model.

    Args:
        model (nnx.Module): The transformer model with encoder and decoder modules.
        batch (Dict[str, jnp.ndarray]): A batch of input data containing:
            - 'encoder_inputs' (jnp.ndarray): Input sequences for the encoder.
        max_seq_len (int, optional): Maximum length of the output sequence to generate. Defaults to 10.

    Returns:
        jnp.ndarray: A tensor of shape `(batch_size, max_seq_len)` containing the predicted token indices.
    """
    if model.model_backbone.__class__.__name__ == 'Transformer':
        preds, output = generate_sequence(model, batch)
    else:
        raise NotImplementedError
    

    return preds
