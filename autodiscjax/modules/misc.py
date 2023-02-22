import autodiscjax as adx
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

class ClampModule(adx.Module):
    low: PyTree = None
    high: PyTree = None

    def __init__(self, out_treedef, out_shape, out_dtype, low=None, high=None):
        super().__init__(out_treedef, out_shape, out_dtype)

        if isinstance(low, float):
            self.low = jtu.tree_map(lambda val, shape, dtype: val * jnp.ones(shape=shape, dtype=dtype),
                                        low, self.out_shape, self.out_dtype)
        else:
            self.low = low

        if isinstance(high, float):
            self.high = jtu.tree_map(lambda val, shape, dtype: val * jnp.ones(shape=shape, dtype=dtype),
                                        high, self.out_shape, self.out_dtype)
        else:
            self.high = high

    @eqx.filter_jit
    def clamp(self, pytree, is_leaf=None):
        return self.clamp_high(self.clamp_low(pytree, is_leaf=is_leaf), is_leaf=is_leaf)

    @eqx.filter_jit
    def clamp_low(self, pytree, is_leaf=None):
        if self.low is not None:
            return jtu.tree_map(lambda val, low: jnp.maximum(val, low), pytree, self.low, is_leaf=is_leaf)
        else:
            return pytree

    @eqx.filter_jit
    def clamp_high(self, pytree, is_leaf=None):
        if self.high is not None:
            return jtu.tree_map(lambda val, high: jnp.minimum(val, high), pytree, self.high, is_leaf=is_leaf)
        else:
            return pytree