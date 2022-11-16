import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree

class BaseModule(eqx.Module):
    out_treedef: jtu.PyTreeDef = eqx.static_field()
    out_shape: PyTree = eqx.static_field()
    out_dtype: PyTree = eqx.static_field()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def out_sanity_check(self, outputs):
        out_flat, out_treedef = jtu.tree_flatten(outputs)
        assert out_treedef == self.out_treedef
        out_shape_flat = jtu.tree_leaves(self.out_shape, is_leaf=lambda node: isinstance(node, tuple))
        out_dtype_flat = jtu.tree_leaves(self.out_dtype)
        for v_idx, v in enumerate(out_flat):
            assert v.shape == out_shape_flat[v_idx]
            assert v.dtype == out_dtype_flat[v_idx]