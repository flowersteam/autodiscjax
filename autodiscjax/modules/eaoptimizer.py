from autodiscjax import DictTree
from autodiscjax.utils.misc import normal
import equinox as eqx
from jax import vmap, jit
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

class EAOptimizer(eqx.Module):
    """
    very simple EA explorer that do random mutations of params and select the best one
    """
    n_optim_steps: int = eqx.static_field()
    n_workers: int = eqx.static_field()
    noise_std: DictTree


    def __call__(self, key, params, loss_fn):
        loss_fn = vmap(jit(loss_fn))

        for optim_step_idx in range(self.n_optim_steps):
            # random mutations
            zero_mean = jtu.tree_map(lambda p: jnp.zeros_like(p), params)
            out_treedef = jtu.tree_structure(params)
            out_shape = jtu.tree_map(lambda p: (self.n_workers,) + p.shape, params)
            out_dtype = jtu.tree_map(lambda p: p.dtype, params)
            key, subkey = jrandom.split(key)
            batched_params = jtu.tree_map(lambda p: jnp.repeat(p[jnp.newaxis], self.n_workers, axis=0), params)
            noise = normal(key, zero_mean, self.noise_std, out_treedef, out_shape, out_dtype)
            worker_params = jtu.tree_map(lambda p, e: p + e, batched_params, noise)

            # evaluate fitness
            subkeys = jrandom.split(key, 1 + self.n_workers)
            key, subkeys = subkeys[0], subkeys[1:]
            losses = loss_fn(subkeys, worker_params)
            losses_flat, _ = jtu.tree_flatten(losses) #shape (n_workers, )
            losses_flat = jnp.concatenate(losses_flat)

            # Select best worker
            params = worker_params[losses_flat.argmin()]

        return params
