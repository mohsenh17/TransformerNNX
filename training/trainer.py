from typing import Tuple, Dict, Optional
from flax import nnx
import optax
import jax.numpy as jnp

def loss_fn(model: nnx.Module, 
            encoder_inputs: jnp.ndarray, 
            decoder_inputs: jnp.ndarray, 
            targets: jnp.ndarray, 
            num_heads: int, 
            mask: Optional[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the loss and logits for a given model and inputs.

    Args:
        model (nnx.Module): The Flax model.
        encoder_inputs (jnp.ndarray): One-hot encoded encoder inputs of shape (batch_size, seq_length, vocab_size).
        decoder_inputs (jnp.ndarray): One-hot encoded decoder inputs of shape (batch_size, seq_length, vocab_size).
        targets (jnp.ndarray): One-hot encoded target labels of shape (batch_size, seq_length, vocab_size).
        num_heads (int): Number of attention heads in the transformer.
        mask (Optional[jnp.ndarray]): Optional attention mask of shape (batch_size, seq_length, seq_length).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - loss: Scalar loss value.
            - logits: Model predictions of shape (batch_size, seq_length, vocab_size).
    """
    logits = model(encoder_inputs, decoder_inputs, num_heads, mask)
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
        num_heads=2, 
        mask=mask
    )
    metrics.update(loss=loss, logits=logits, labels=batch['targets'])
    optimizer.update(grads)
    return loss, nnx.sigmoid(logits)

@nnx.jit
def eval_step(model: nnx.Module, 
              metrics: Dict, 
              batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    not Implemented
    """
    raise NotImplementedError
    loss, logits = loss_fn(
        model, 
        encoder_inputs=batch['inputs'], 
        decoder_inputs=batch['targets'], 
        targets=batch['targets'], 
        num_heads=2, 
        mask=None
    )
    metrics.update(loss=loss, logits=logits, labels=batch['inputs'])
    return loss

@nnx.jit
def pred_step(model: nnx.Module, 
              batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    not Implemented
    """
    raise NotImplementedError
    logits = model(batch['features'])
    return nnx.sigmoid(logits)
