from autodiscjax import DictTree
from autodiscjax.modules.imgepwrappers import HypercubeGoalGenerator, IMFlowGoalGenerator, LearningProgressIM, NearestNeighborInterventionSelector, FilterGoalEmbeddingEncoder
from jax import vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

def test_hypercube_goal_generator():
    key = jrandom.PRNGKey(0)

    batch_size = 50
    gs_ndim = 2
    goal_embedding_tree = "placeholder"
    goal_embedding_treedef = jtu.tree_structure(goal_embedding_tree)
    goal_embedding_shape = jtu.tree_map(lambda _: (gs_ndim,), goal_embedding_tree)
    goal_embedding_dtype = jtu.tree_map(lambda _: jnp.float32, goal_embedding_tree)
    low = 0.
    high = None
    hypercube_scaling = 1.5

    hypercube_generator = HypercubeGoalGenerator(goal_embedding_treedef, goal_embedding_shape, goal_embedding_dtype,
                                                 low, high, hypercube_scaling)
    hypercube_generator = vmap(hypercube_generator, in_axes=(0, None, None, None), out_axes=0)
    target_goal_embedding_library = None

    key, subkey = jrandom.split(key)
    S = jnp.array([[1., 0.5], [0.5, 0.05]])
    mu = jnp.array([2, 2])
    reached_goal_embedding_library = jrandom.normal(subkey, shape=(batch_size, gs_ndim)) @ S + mu
    reached_goal_embedding_library = reached_goal_embedding_library @ S

    key, *subkeys = jrandom.split(key, num=batch_size + 1)
    next_goals, log_data = hypercube_generator(jnp.array(subkeys), target_goal_embedding_library,
                                            reached_goal_embedding_library, None)

    # Assert that all next goals are on the upper-right side of the grid
    assert next_goals.shape == (batch_size, gs_ndim)
    assert (next_goals > low).all()
    hypercube_size = (reached_goal_embedding_library.max(0)-reached_goal_embedding_library.min(0))
    hypercube_center = reached_goal_embedding_library.min(0) + hypercube_size/2.0
    hypercube_low = hypercube_center - hypercube_size*hypercube_scaling/2.0
    hypercube_high = hypercube_center + hypercube_size*hypercube_scaling/2.0
    assert (next_goals > hypercube_low).all() and (next_goals < hypercube_high).all()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(*reached_goal_embedding_library.transpose(), label="reached goals")
    plt.scatter(*next_goals.transpose(), label="next goals")
    plt.legend()
    plt.show()

def test_im_flow_goal_generator():
    batch_size = 50
    gs_ndim = 2
    goal_embedding_tree = "placeholder"
    goal_embedding_treedef = jtu.tree_structure(goal_embedding_tree)
    goal_embedding_shape = jtu.tree_map(lambda _: (gs_ndim,), goal_embedding_tree)
    goal_embedding_dtype = jtu.tree_map(lambda _: jnp.float32, goal_embedding_tree)

    IM_fn = LearningProgressIM()
    IM_val_scaling = 10.
    IM_grad_scaling = 0.
    random_proba = 0.
    flow_noise = 0.
    time_window = jnp.r_[-batch_size:0]

    imflow_generator = IMFlowGoalGenerator(goal_embedding_treedef, goal_embedding_shape, goal_embedding_dtype, None, None,
                                           IM_fn, IM_val_scaling, IM_grad_scaling, random_proba, flow_noise, time_window)
    imflow_generator = vmap(imflow_generator, in_axes=(0, None, None, None), out_axes=0)

    key = jrandom.PRNGKey(0)
    subkeys = jrandom.split(key, num=5)
    # Target goals:
    # first batch is empty
    # second batch: first half is on the upper-right corner, other half is on lower-right corner
    target_goal_embedding_library = jnp.concatenate([jnp.empty(shape=(batch_size, gs_ndim)),
                                                     jrandom.uniform(subkeys[0], shape=(batch_size//2, gs_ndim), minval=jnp.array([.8, .75]), maxval=jnp.array([1., 1.])),
                                                     jrandom.uniform(subkeys[1], shape=(batch_size//2, gs_ndim), minval=jnp.array([.8, 0.]), maxval=jnp.array([1., .25]))])
    # Reached goals:
    # first batch is randomly sampled in middle band of the grid
    # second batch: first half gets closer on the upper-right way and other half gets further on the upper-left way
    reached_goal_embedding_library = jnp.concatenate([jrandom.uniform(subkeys[2], shape=(batch_size, gs_ndim),
                                                                     minval=jnp.array([.4, 0.]), maxval=jnp.array([.6, .5])),
                                                     jrandom.uniform(subkeys[3], shape=(batch_size // 2, gs_ndim),
                                                                     minval=jnp.array([.6, .5]), maxval=jnp.array([.8, .75])),
                                                     jrandom.uniform(subkeys[4], shape=(batch_size // 2, gs_ndim),
                                                                     minval=jnp.array([.2, .5]), maxval=jnp.array([.4, .75]))])
    key, *subkeys = jrandom.split(key, num=batch_size + 1)
    next_goals, log_data = imflow_generator(jnp.array(subkeys), target_goal_embedding_library, reached_goal_embedding_library, None)

    # Assert that all next goals are on the upper-right side of the grid
    assert next_goals.shape == (batch_size, gs_ndim)
    assert ((next_goals > 0.5).all(-1).sum() / len(next_goals)) > 0.8

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(*target_goal_embedding_library[batch_size:].transpose(), label="target goals")
    plt.scatter(*reached_goal_embedding_library.transpose(), label="reached goals")
    plt.scatter(*next_goals.transpose(), label="next goals")
    plt.legend()
    plt.show()

def test_nn_intervention_selector():
    key = jrandom.PRNGKey(0)

    batch_size = 10
    intervention_selector_tree = "placeholder"
    intervention_selector_treedef = jtu.tree_structure(intervention_selector_tree)
    intervention_selector_shape = jtu.tree_map(lambda _: (), intervention_selector_tree)
    intervention_selector_dtype = jtu.tree_map(lambda _: jnp.int32, intervention_selector_tree)
    k = 1
    gc_intervention_selector = NearestNeighborInterventionSelector(intervention_selector_treedef,
                                                                         intervention_selector_shape,
                                                                         intervention_selector_dtype, k)
    gc_intervention_selector = vmap(gc_intervention_selector, in_axes=(0, 0, None, None), out_axes=(0, None))

    key, *subkeys = jrandom.split(key, num=batch_size + 1)
    target_goals_embeddings = jnp.arange(10).reshape((10,1))
    reached_goal_embedding_library = jnp.flip(target_goals_embeddings+0.1, axis=0)
    source_interventions_ids, log_data = gc_intervention_selector(jnp.array(subkeys), target_goals_embeddings,
                                                                          reached_goal_embedding_library,
                                                                          None)
    assert (source_interventions_ids == jnp.flip(jnp.arange(10), axis=0)).all()

def test_filter_goal_embedding_encoder():
    key = jrandom.PRNGKey(0)

    goal_embedding_tree = "placeholder"
    goal_embedding_treedef = jtu.tree_structure(goal_embedding_tree)
    goal_embedding_shape = jtu.tree_map(lambda _: (3, ),
                                        goal_embedding_tree)
    goal_embedding_dtype = jtu.tree_map(lambda _: jnp.float32, goal_embedding_tree)
    goal_filter_fn = jtu.Partial(
        lambda outputs: jnp.array([outputs.a[-1], outputs.b[-1], outputs.c[-1]]))
    goal_embedding_encoder = FilterGoalEmbeddingEncoder(goal_embedding_treedef, goal_embedding_shape,
                                                              goal_embedding_dtype, goal_filter_fn)

    outputs = DictTree(a=jnp.arange(10), b=jnp.arange(10,20), c=jnp.arange(20,30))

    key, subkey = jrandom.split(key)
    filtered_ouputs, log_data = goal_embedding_encoder(subkey, outputs)
    assert (filtered_ouputs == jnp.array([9, 19, 29])).all()