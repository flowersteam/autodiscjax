import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree


def get_full(pytree: PyTree):
    return pytree



def merge_replace(old_val: PyTree, new_val: PyTree, enforce_same_treedef: bool = True):
    if enforce_same_treedef:
        assert jtu.tree_structure(old_val) == jtu.tree_structure(new_val)
    return new_val


def merge_concatenate(old_val: PyTree, new_val: PyTree, axis=0):
    return jtu.tree_map(lambda x_old, x_new:  jnp.concatenate([x_old, x_new], axis=axis), old_val, new_val)

def merge_stack(old_val: PyTree, new_val: PyTree, axis=0):
    return jtu.tree_map(lambda x_old, x_new:  jnp.stack([x_old, x_new], axis=axis), old_val, new_val)
