from autodiscjax import DictTree
import equinox as eqx
from jax import jit, value_and_grad
import jax.random as jrandom
import jax.tree_util as jtu
import optax
import time

class SGDOptimizer(eqx.Module):
    n_optim_steps: int = eqx.static_field()
    optimizer: optax._src.base.GradientTransformation


    def __init__(self, n_optim_steps, lr):
        self.n_optim_steps = n_optim_steps
        lr_flat, lr_treedef = jtu.tree_flatten(lr)
        leaves_ids = list(range(lr_treedef.num_leaves))
        param_labels = lr_treedef.unflatten(leaves_ids)
        transforms = {}
        for leaf_idx, leaf_lr in zip(leaves_ids, lr_flat):
            transforms[leaf_idx] = optax.adam(leaf_lr)
        self.optimizer = optax.multi_transform(transforms, param_labels.to_dict())

    #@jit
    def __call__(self, key, params, loss_fn):

        opt_state = self.optimizer.init(params.to_dict())

        for optim_step_idx in range(self.n_optim_steps):
            tstart = time.time()
            key, subkey = jrandom.split(key)
            loss, grads = value_and_grad(jit(loss_fn), argnums=1)(subkey, params)

            # TODO: remove the DictTree to dict conversion if addict issue mewwts/addict#150 gets solved
            params = params.to_dict()
            updates, opt_state = self.optimizer.update(grads.to_dict(), opt_state)
            params = optax.apply_updates(params, updates)
            params = DictTree(params)

            tend = time.time()
            print(loss, tend-tstart)

        return params
