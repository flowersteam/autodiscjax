import autodiscjax as adx
import equinox as eqx
from autodiscjax.utils.accessors import merge_concatenate
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import os

def run_imgep_evaluation(jax_platform_name: str, seed: int, n_perturbations: int, save_folder: str,
                         intervention_params_library: adx.DictTree, intervention_fn: eqx.Module,
                         perturbation_generator: eqx.Module, perturbation_fn: eqx.Module,
                         system_rollout: eqx.Module, rollout_statistics_encoder: eqx.Module,
                         goal_embedding_encoder: eqx.Module,
                         out_sanity_check=True):
    
    # Set platform device
    jax.config.update("jax_platform_name", jax_platform_name)

    # Set random seed
    key = jrandom.PRNGKey(seed)

    # Initialize History
    history = adx.DictTree()
    history.perturbation_params_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(shape[0], 0, ) + shape[1:], dtype=dtype),
                                                       perturbation_generator.out_shape, perturbation_generator.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    history.system_output_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(shape[0], 0, ) + shape[1:], dtype=dtype),
                                                       system_rollout.out_shape, system_rollout.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    history.reached_goal_embedding_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(shape[0], 0, ) + shape[1:], dtype=dtype),
                                                       goal_embedding_encoder.out_shape, goal_embedding_encoder.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    history.system_rollout_statistics_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(shape[0], 0, ) + shape[1:], dtype=dtype),
                                                       rollout_statistics_encoder.out_shape, rollout_statistics_encoder.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    
    # Run Evaluation
    for perturbation_idx in range(n_perturbations):

        # generate perturbation
        key, subkey = jrandom.split(key)
        perturbations_params = perturbation_generator(subkey)
        if out_sanity_check:
            perturbation_generator.out_sanity_check(perturbations_params)

        # rollout system
        key, subkey = jrandom.split(key)
        system_outputs = system_rollout(subkey, intervention_fn, intervention_params_library, perturbation_fn, perturbations_params)
        if out_sanity_check:
            system_rollout.out_sanity_check(system_outputs)

        # represent outputs -> goals
        key, subkey = jrandom.split(key)
        reached_goals_embeddings = goal_embedding_encoder(subkey, system_outputs)
        if out_sanity_check:
            goal_embedding_encoder.out_sanity_check(reached_goals_embeddings)

        # represent outputs -> other statistics
        key, subkey = jrandom.split(key)
        system_rollouts_statistics = rollout_statistics_encoder(subkey, system_outputs)
        if out_sanity_check:
            rollout_statistics_encoder.out_sanity_check(system_rollouts_statistics)


        # Append to history
        perturbations_params = jtu.tree_map(lambda val: val[:, jnp.newaxis], perturbations_params)
        system_outputs = jtu.tree_map(lambda val: val[:, jnp.newaxis], system_outputs)
        reached_goals_embeddings = jtu.tree_map(lambda val: val[:, jnp.newaxis], reached_goals_embeddings)
        system_rollouts_statistics = jtu.tree_map(lambda val: val[:, jnp.newaxis], system_rollouts_statistics)
        history = history.update_node("perturbation_params_library", perturbations_params, merge_concatenate, axis=1)
        history = history.update_node("system_output_library", system_outputs, merge_concatenate, axis=1)
        history = history.update_node("reached_goal_embedding_library", reached_goals_embeddings, merge_concatenate, axis=1)
        history = history.update_node("system_rollout_statistics_library", system_rollouts_statistics, merge_concatenate, axis=1)


    # Save history and modules
    history.save(os.path.join(save_folder, "evaluation_history.pickle"), overwrite=True)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "perturbation_generator.eqx"), perturbation_generator)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "perturbation_fn.eqx"), perturbation_fn)