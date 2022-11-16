from functools import partial
from jax import jit, lax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import PyTree
from typing import Callable

@partial(jit, static_argnums=(1, 2))
def filter(pytree: PyTree, filter_fn: Callable, out_treedef: jtu.PyTreeDef):
        sub_pytree = filter_fn(pytree)
        out_flat, _ = jtu.tree_flatten(sub_pytree)
        return out_treedef.unflatten(out_flat)

@partial(jit, static_argnums=(1, 2, 3))
def filter_update(pytree: PyTree, filter_fn: Callable, update_fn: Callable, out_treedef: jtu.PyTreeDef):
    sub_pytree = filter_fn(pytree)
    updated_sub_pytree = update_fn(sub_pytree)
    out_flat, _ = jtu.tree_flatten(updated_sub_pytree)
    return out_treedef.unflatten(out_flat)


def uniform(key, low: PyTree, high: PyTree, out_treedef: jtu.PyTreeDef, out_shape: PyTree, out_dtype: PyTree):
    key = out_treedef.unflatten(jrandom.split(key, out_treedef.num_leaves))
    return jtu.tree_map(
        lambda key, low, high, shape, dtype: low + (high - low) * jrandom.uniform(key, shape=shape, dtype=dtype),
        key, low, high, out_shape, out_dtype)


def normal(key, mean: PyTree, std: PyTree, out_treedef: jtu.PyTreeDef, out_shape: PyTree, out_dtype: PyTree):
    key = out_treedef.unflatten(jrandom.split(key, out_treedef.num_leaves))
    return jtu.tree_map(
        lambda key, mean, std, shape, dtype: mean + std * jrandom.normal(key, shape=shape, dtype=dtype),
        key, mean, std, out_shape, out_dtype)

@partial(jit, static_argnums=(2, ))
def nearest_neighbors(Y, X, k):
    """
    Arguments:
    Y: Array [Ny, D]
    X: Array[Nx, D
    k: int

    Returns:
    X_nearest_ids: Array[Ny, k] - matrix of the  k closest point ids in X (for each target point in Y),
    distances: Array[Ny, k] - corresponding distances
    """
    Ny, Dy = Y.shape
    Ny, Dx = X.shape
    assert Dy == Dx, "Points in X and Y must lie in the same D-dimensional space"

    distance_matrix = jnp.sum((Y[:, jnp.newaxis, :] - X[jnp.newaxis, :, :])**2, axis=-1)
    nearest_distances_reverse, X_nearest_ids = lax.top_k(jnp.reciprocal(distance_matrix), k)
    return X_nearest_ids, jnp.reciprocal(nearest_distances_reverse)

@jit
def hardplus(x):
    return jnp.maximum(x, 0.)

@jit
def softplus(x):
    return jnp.log(1.0 + jnp.exp(x))*100.0

@jit
def inv_softplus(x):
    return jnp.log(jnp.exp(x/100.0) - 1.0)