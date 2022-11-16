from addict import Dict
from autodiscjax.utils.accessors import get_full, merge_replace
import equinox as eqx
from jax.util import unzip2
import jax.tree_util as jtu
from jaxtyping import PyTree
from pathlib import Path
import pickle
from typing import Union

@jtu.register_pytree_node_class
class DictTree(Dict):

    def tree_flatten(self):
        return unzip2(sorted(self.items()))[::-1]

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(zip(aux_data, children))

    def save(self, path: Union[str, Path], overwrite: bool = False):
        path = Path(path)
        if path.suffix != ".pickle":
            path = path.with_suffix(".pickle")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise RuntimeError(f'File {path} already exists.')
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: Union[str, Path]):
        path = Path(path)
        if not path.is_file():
            raise ValueError(f'Not a file: {path}')
        if path.suffix != ".pickle":
            raise ValueError(f'Not a {".pickle"} file: {path}')
        with open(path, 'rb') as file:
            dictree = pickle.load(file)
        return dictree

    def get_node(self, node_path: str, get_fn=get_full):
        """
        node path: str of the form key1.key2....
        """
        return get_fn(eval(f"self.{node_path}"))


    def update_node(self, node_path: str, new_val: PyTree, update_fn=merge_replace, **update_fn_params):
        """
        Functional update that returns the modified dictree (not an in-place mutation as this goes against JAX functional paradigm principles).
        """
        old_val = self.get_node(node_path)
        updated_val = update_fn(old_val, new_val, **update_fn_params)
        new_dictree = eqx.tree_at(lambda dictree: dictree.get_node(node_path), self, updated_val)
        return new_dictree