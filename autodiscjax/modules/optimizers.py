import autodiscjax as adx
from autodiscjax.utils.accessors import merge_concatenate
from autodiscjax.utils.misc import normal
import equinox as eqx
from jax import jit, value_and_grad, vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import PyTree
import optax
import time

class BaseOptimizer(adx.Module):
    low: PyTree
    high: PyTree

    def __call__(self, key, params, loss_fn):
        raise NotImplementedError

    @eqx.filter_jit
    def clamp(self, out_pytree, is_leaf=None):
        return jtu.tree_map(lambda val, low, high: jnp.minimum(jnp.maximum(val, low), high), out_pytree, self.low,
                            self.high, is_leaf=is_leaf)


class EAOptimizer(BaseOptimizer):
    """
    very simple EA explorer that do random mutations of params and select the best one
    """
    n_optim_steps: int = eqx.static_field()
    n_workers: int = eqx.static_field()
    noise_std: adx.DictTree


    def __call__(self, key, params, loss_fn):
        loss_fn = vmap(jit(loss_fn))

        log_data = adx.DictTree()
        log_data.train_loss = jnp.empty(shape=(0, self.n_workers), dtype=jnp.float32)
        log_data.trainstep_time = jnp.empty(shape=(0, ), dtype=jnp.float32)

        for optim_step_idx in range(self.n_optim_steps):
            step_start = time.time()
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
            losses_flat, _ = jtu.tree_flatten(losses)
            losses_flat = jnp.concatenate(losses_flat) #shape (n_workers, )

            # Select best worker
            best_worker_idx = losses_flat.argmin()
            params = jtu.tree_map(lambda p: p[best_worker_idx], worker_params)

            # Clamp params
            params = self.clamp(params)

            step_end = time.time()
            log_data = log_data.update_node("train_loss", losses_flat[jnp.newaxis], merge_concatenate)
            log_data = log_data.update_node("trainstep_time", jnp.array([step_end-step_start]), merge_concatenate)

        return params, log_data


class SGDOptimizer(BaseOptimizer):
    n_optim_steps: int = eqx.static_field()
    optimizer: optax._src.base.GradientTransformation

    def __init__(self, out_treedef, out_shape, out_dtype, low, high, n_optim_steps, lr):
        super().__init__(out_treedef, out_shape, out_dtype, low, high)
        self.n_optim_steps = n_optim_steps
        lr_flat, lr_treedef = jtu.tree_flatten(lr)
        leaves_ids = list(range(lr_treedef.num_leaves))
        param_labels = lr_treedef.unflatten(leaves_ids)
        transforms = {}
        for leaf_idx, leaf_lr in zip(leaves_ids, lr_flat):
            transforms[leaf_idx] = optax.adam(leaf_lr)

        if isinstance(param_labels, adx.DictTree):
            param_labels = param_labels.to_dict()
        self.optimizer = optax.multi_transform(transforms, param_labels)

    #@eqx.filter_jit  => much longer
    def value_and_grad(self, key, params, loss_fn):
        return value_and_grad(loss_fn, argnums=1)(key, params)


    def __call__(self, key, params, loss_fn):
        # TODO:
        #  remove the adx.DictTree to dict conversion if addict issue mewwts/addict#150 gets solved
        #  scan loop

        if isinstance(params, adx.DictTree):
            params = params.to_dict()
            is_params_dictree = True
        else:
            is_params_dictree = False

        opt_state = self.optimizer.init(params)

        log_data = adx.DictTree()
        log_data.train_loss = jnp.empty(shape=(0, ), dtype=jnp.float32)
        log_data.trainstep_time = jnp.empty(shape=(0, ), dtype=jnp.float32)

        for optim_step_idx in range(self.n_optim_steps):
            step_start = time.time()

            key, subkey = jrandom.split(key)
            if is_params_dictree:
                params = adx.DictTree(params)
                # Clamp params
                params = self.clamp(params)
            loss, grads = self.value_and_grad(subkey, params, jit(loss_fn))

            if is_params_dictree:
                params = params.to_dict()
                grads = grads.to_dict()
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            step_end = time.time()
            log_data = log_data.update_node("train_loss", loss[jnp.newaxis], merge_concatenate)
            log_data = log_data.update_node("trainstep_time", jnp.array([step_end - step_start]), merge_concatenate)

        if is_params_dictree:
            params = adx.DictTree(params)
            params = self.clamp(params)

        return params, log_data


class OpenESOptimizer(SGDOptimizer):
    """
    Reference: Salimans et al. (2017) - https://arxiv.org/pdf/1703.03864.pdf
    """
    n_workers: int = eqx.static_field()
    noise_std: adx.DictTree

    def __init__(self, out_treedef, out_shape, out_dtype, low, high, n_optim_steps: int, lr: adx.DictTree, n_workers: int, noise_std: adx.DictTree):
        """
        Args:
            n_workers: int
            noise_std: adx.DictTree with same structure as params
        """
        super().__init__(out_treedef, out_shape, out_dtype, low, high, n_optim_steps, lr)
        self.n_workers = n_workers
        self.noise_std = noise_std

    #@eqx.filter_jit => much longer
    def value_and_grad(self, key, params, loss_fn):
        zero_mean = jtu.tree_map(lambda p: jnp.zeros_like(p), params)
        out_treedef = jtu.tree_structure(params)
        out_shape = jtu.tree_map(lambda p: (self.n_workers,) + p.shape, params)
        out_dtype = jtu.tree_map(lambda p: p.dtype, params)

        key, subkey = jrandom.split(key)
        batched_params = jtu.tree_map(lambda p: jnp.repeat(p[jnp.newaxis], self.n_workers, axis=0), params)
        noise = normal(key, zero_mean, self.noise_std, out_treedef, out_shape, out_dtype)
        worker_params = jtu.tree_map(lambda p, e: p+e, batched_params, noise)
        subkeys = jrandom.split(key, 1+self.n_workers)
        key, subkeys = subkeys[0], subkeys[1:]
        losses = vmap(loss_fn)(subkeys, worker_params)
        losses = jtu.tree_map(lambda node: losses, self.noise_std)
        epsilons = jtu.tree_map(lambda p_new, p, sigma: (p_new - p) / sigma, worker_params, params, self.noise_std)
        grads = jtu.tree_map(lambda l, eps, sigma: (l.reshape((len(l), ) + (1, ) * len(eps.shape[1:])) * eps).sum(0) / (self.n_workers * sigma), losses, epsilons, self.noise_std)

        return loss_fn(key, params), grads