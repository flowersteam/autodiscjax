from autodiscjax.modules.imgepwrappers import IMFlowGoalGenerator, LearningProgressIM
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

def test_im_flow_goal_generator():
    batch_size = 50
    gs_ndim = 2
    goal_embedding_tree = "placeholder"
    goal_embedding_treedef = jtu.tree_structure(goal_embedding_tree)
    goal_embedding_shape = jtu.tree_map(lambda _: (gs_ndim,), goal_embedding_tree)
    batched_goal_embedding_shape = jtu.tree_map(lambda shape: (batch_size,) + shape, goal_embedding_shape,
                                                is_leaf=lambda node: isinstance(node, tuple))
    goal_embedding_dtype = jtu.tree_map(lambda _: jnp.float32, goal_embedding_tree)

    IM_fn = LearningProgressIM()
    IM_grad_scaling = 0.4
    random_popsize = 0.
    selected_popsize = 0.2
    flow_noise = 0.05

    imflow_generator = IMFlowGoalGenerator(goal_embedding_treedef, batched_goal_embedding_shape, goal_embedding_dtype,
                                           IM_fn, IM_grad_scaling, random_popsize, selected_popsize, flow_noise)

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
                                                                     minval=jnp.array([.4, .25]), maxval=jnp.array([.6, .75])),
                                                     jrandom.uniform(subkeys[3], shape=(batch_size // 2, gs_ndim),
                                                                     minval=jnp.array([.6, .5]), maxval=jnp.array([.8, .75])),
                                                     jrandom.uniform(subkeys[4], shape=(batch_size // 2, gs_ndim),
                                                                     minval=jnp.array([.2, .5]), maxval=jnp.array([.4, .75]))])

    next_goals = imflow_generator(key, target_goal_embedding_library, reached_goal_embedding_library)

    # Assert that all next goals are on the upper-right side of the grid
    assert next_goals.shape == (batch_size, gs_ndim)
    assert (next_goals > jnp.array([0.4, 0.4])).all()

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(*target_goal_embedding_library[batch_size:].transpose(), label="target goals")
    # plt.scatter(*reached_goal_embedding_library.transpose(), label="reached goals")
    # plt.scatter(*next_goals.transpose(), label="next goals")
    # plt.legend()
    # plt.show()