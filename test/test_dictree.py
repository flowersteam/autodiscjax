import jax.tree_util as jtu
from autodiscjax.dictree import DictTree

def test_dictree():
    def get_structure(structured):
        flat, tree = jtu.tree_flatten(structured)
        unflattened = jtu.tree_unflatten(tree, flat)
        return flat, tree, unflattened

    history = DictTree()
    history.a.a2 = 2
    history.b = 2
    flat, tree, unflattened = get_structure(history)

    dict_history = dict()
    dict_history["a"] = dict()
    dict_history["a"]["a2"] = 2
    dict_history["b"] = 2
    dict_flat, dict_tree, dict_unflattened = get_structure(dict_history)

    assert flat == dict_flat
    assert unflattened == dict_unflattened

