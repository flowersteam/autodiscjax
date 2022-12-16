from autodiscjax.modules.imgepwrappers import IMFlowGoalGenerator, LearningProgressIM
from jax import vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

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