import jax
import orbax.checkpoint as ocp
from flax import nnx


def save_model(model: nnx.Module, ckpt_dir: str) -> None:
    """
    Processes the model state, modifies the PRNG key, and saves the state and keys to a checkpoint directory.

    Args:
        model (Any): The model whose state is being processed.
        ckpt_dir (str): Directory where the checkpoint files will be saved.

    **Note:** 
        As of December 2024, Orbax won't handle 'key<fry>' as a data type hence the key needs to be converted. 
    """
    # Retrieve and split the model's state
    keys, state = nnx.state(model, nnx.RngKey, ...)
    keys = jax.tree_map(jax.random.key_data, keys)

    # Create a new empty checkpoint directory
    ckpt_dir = ocp.test_utils.erase_and_create_empty(ckpt_dir)

    # Initialize the checkpointing system and save the state
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(f'{ckpt_dir}/state', state)
    checkpointer.save(f'{ckpt_dir}/keys', keys)


def restore_model(model_init: nnx.Module, ckpt_dir: str) -> nnx.Module:
    """
    Restores a model from a checkpoint.

    Args:
        model_init (Any): The initialized model object to be restored.
        ckpt_dir (str): Directory containing the checkpoint files.

    Returns:
        Any: The restored model object with updated state.

    **Note:** 
        As of December 2024, Orbax won't handle 'key<fry>' as a data type hence the key needs to be converted. 
    
    """
    # Evaluate the abstract shape of the model and split into graph/state
    abstract_model = nnx.eval_shape(lambda: model_init)
    abstract_keys, abstract_state = nnx.state(abstract_model, nnx.RngKey, ...)

    # Restore the state and key from checkpoint
    checkpointer = ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(f"{ckpt_dir}/state", abstract_state)
    restored_keys = checkpointer.restore(f"{ckpt_dir}/keys", abstract_keys)

    # Update the state with the restored key
    restored_keys = jax.tree_map(jax.random.wrap_key_data, restored_keys)
    nnx.update(model_init, restored_keys, restored_state)
        
    return model_init
