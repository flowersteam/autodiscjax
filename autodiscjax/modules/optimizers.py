import autodiscjax as adx
from autodiscjax.modules.misc import ClampModule
from autodiscjax.utils.misc import normal
import equinox as eqx
from jax import jit, value_and_grad, vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import PyTree
import optax
from typing import Callable

class BaseOptimizer(ClampModule):
    n_optim_steps: int = eqx.static_field()
    n_workers: int = eqx.static_field()

    def __init__(self, out_treedef, out_shape, out_dtype, low, high, n_optim_steps, n_workers):
        super().__init__(out_treedef, out_shape, out_dtype, low, high)
        self.n_optim_steps = n_optim_steps
        self.n_workers = n_workers

    def __call__(self, key, params, evaluate_worker_fn):
        """
        evaluate_worker_fn: key, params -> loss, log_data
        """
        raise NotImplementedError


class EAOptimizer(BaseOptimizer):
    init_noise_std: float
    schedule_fn: Callable

    def __init__(self, out_treedef, out_shape, out_dtype, low, high, n_optim_steps, n_workers, init_noise_std):
        super().__init__(out_treedef, out_shape, out_dtype, low, high, n_optim_steps, n_workers)
        self.init_noise_std = init_noise_std
        self.schedule_fn = jtu.Partial(optax.polynomial_schedule(init_value=1.0, end_value=0., power=1, transition_steps=n_optim_steps))


    def update_worker(self, key, worker_params, optim_step_idx):
        zero_mean = jtu.tree_map(lambda p: jnp.zeros_like(p), worker_params)
        noise_std = jtu.tree_map(lambda init_noise_std: self.schedule_fn(optim_step_idx)*init_noise_std, self.init_noise_std)
        noise = normal(key, zero_mean, noise_std, self.out_treedef, self.out_shape, self.out_dtype)
        worker_params = jtu.tree_map(lambda p, e: p + e, worker_params, noise)
        return worker_params


    def __call__(self, key, params, evaluate_worker_fn):
        update_workers = vmap(self.update_worker, in_axes=(0, 0, None), out_axes=0)
        evaluate_workers = vmap(evaluate_worker_fn, in_axes=(0, 0), out_axes=(0, 0))
        clamp_workers = vmap(self.clamp, in_axes=0, out_axes=0)

        log_data = []
        selected_worker_idx = 0

        for optim_step_idx in range(self.n_optim_steps):
            # Replicate best worker params
            workers_params = jtu.tree_map(lambda p: jnp.repeat(p[jnp.newaxis], self.n_workers, axis=0), params)

            # Update worker params with random mutations
            key, *subkeys = jrandom.split(key, num=self.n_workers + 1)
            workers_params = update_workers(jnp.array(subkeys), workers_params, optim_step_idx)

            # Clamp params
            workers_params = clamp_workers(workers_params)

            # Evaluate fitness
            key, *subkeys = jrandom.split(key, num=self.n_workers + 1)
            losses, evaluate_log_data = evaluate_workers(jnp.array(subkeys), workers_params)

            # append worker params and source ids to log
            if isinstance(evaluate_log_data, adx.DictTree):
                evaluate_log_data.workers_params = workers_params
                evaluate_log_data.source_workers_ids = jnp.array([selected_worker_idx]*self.n_workers)
            log_data.append(evaluate_log_data)

            # Select best worker
            selected_worker_idx = losses.argmin()
            params = jtu.tree_map(lambda p: p[selected_worker_idx], workers_params)

        return params, log_data


class SGDOptimizer(BaseOptimizer):
    init_noise_std: PyTree
    lr: PyTree

    def __init__(self, out_treedef, out_shape, out_dtype, low, high, n_optim_steps, n_workers, init_noise_std, lr):
        super().__init__(out_treedef, out_shape, out_dtype, low, high, n_optim_steps, n_workers)
        self.init_noise_std = init_noise_std
        self.lr = lr

    def init_worker(self, key, params):
        zero_mean = jtu.tree_map(lambda p: jnp.zeros_like(p), params)
        noise = normal(key, zero_mean, self.init_noise_std, self.out_treedef, self.out_shape, self.out_dtype)
        params = jtu.tree_map(lambda p, e: p + e, params, noise)
        return params

    @eqx.filter_jit
    def update_worker(self, params, grads, optimizer, opt_state):
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, optimizer, opt_state

    def call_worker(self, key, params, evaluate_worker_fn):

        # Prepare optizer
        lr_flat, lr_treedef = jtu.tree_flatten(self.lr)
        leaves_ids = list(range(lr_treedef.num_leaves))
        param_labels = lr_treedef.unflatten(leaves_ids)
        transforms = {}
        for leaf_idx, leaf_lr in zip(leaves_ids, lr_flat):
            transforms[leaf_idx] = optax.adam(leaf_lr)

        # Init params
        key, subkey = jrandom.split(key)
        params = self.init_worker(subkey, params)
        if isinstance(param_labels, adx.DictTree):
            param_labels = param_labels.to_dict()

        if isinstance(params, adx.DictTree):
            params = params.to_dict()
            is_params_dictree = True
        else:
            is_params_dictree = False


        # Init optimizer
        optimizer = optax.multi_transform(transforms, param_labels)
        opt_state = optimizer.init(params)

        log_data = []

        for optim_step_idx in range(self.n_optim_steps):

            if is_params_dictree:
                params = adx.DictTree(params)
                # Clamp params
                params = self.clamp(params)

            key, subkey = jrandom.split(key)
            (loss, evaluate_log_data), grads = value_and_grad(evaluate_worker_fn, argnums=1, has_aux=True)(subkey, params)

            if is_params_dictree:
                params = params.to_dict()
                grads = grads.to_dict()

            updated_params, optimizer, opt_state = self.update_worker(params, grads, optimizer, opt_state)

            # append worker params and source ids to log
            if isinstance(evaluate_log_data, adx.DictTree):
                evaluate_log_data.workers_params = params
                evaluate_log_data.source_workers_ids = jnp.array([0])
            log_data.append(evaluate_log_data)

            params = updated_params

        if is_params_dictree:
            params = adx.DictTree(params)
            params = self.clamp(params)

        return params, loss, log_data


    def __call__(self, key, params, evaluate_worker_fn):

        # Replicate worker params
        workers_params = jtu.tree_map(lambda p: jnp.repeat(p[jnp.newaxis], self.n_workers, axis=0), params)

        key, *subkeys = jrandom.split(key, num=self.n_workers+1)
        workers_params, losses, log_data = vmap(self.call_worker, in_axes=(0, 0, None), out_axes=(0, 0, 0))(jnp.array(subkeys), workers_params, evaluate_worker_fn)

        # Select best worker
        selected_worker_idx = losses.argmin()
        params = jtu.tree_map(lambda p: p[selected_worker_idx], workers_params)

        return params, log_data