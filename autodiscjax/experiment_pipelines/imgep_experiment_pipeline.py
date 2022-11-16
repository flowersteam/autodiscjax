import autodiscjax as adx
import equinox as eqx
from autodiscjax.utils.accessors import merge_concatenate
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import os

def run_imgep_experiment(jax_platform_name: str, seed: int, n_random_batches: int, n_imgep_batches: int, save_folder: str,
                         random_intervention_generator: eqx.Module, intervention_fn: eqx.Module,
                         perturbation_generator: eqx.Module, perturbation_fn: eqx.Module,
                         system_rollout: eqx.Module, rollout_statistics_encoder: eqx.Module,
                         goal_generator: eqx.Module, gc_intervention_selector: eqx.Module,
                         gc_intervention_optimizer: eqx.Module,
                         goal_embedding_encoder: eqx.Module,
                         goal_achievement_loss: eqx.Module,
                         out_sanity_check=True):
    
    # Set platform device
    jax.config.update("jax_platform_name", jax_platform_name)

    # Set random seed
    key = jrandom.PRNGKey(seed)


    if out_sanity_check:
        assert (goal_generator.out_treedef == goal_embedding_encoder.out_treedef) \
               and (goal_generator.out_shape == goal_embedding_encoder.out_shape) \
               and (goal_generator.out_dtype == goal_embedding_encoder.out_dtype),  \
            "goal generator and goal encoder must operate in same spaces"

        assert (random_intervention_generator.out_treedef == gc_intervention_optimizer.out_treedef) \
               and (random_intervention_generator.out_shape == gc_intervention_optimizer.out_shape) \
               and (random_intervention_generator.out_dtype == gc_intervention_optimizer.out_dtype),  \
            "random intervention generator and goal-conditionned intervention operator must operate in same spaces"

    # Initialize History
    history = adx.DictTree()
    history.target_goal_embedding_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape[1:], dtype=dtype),
                                                         goal_generator.out_shape, goal_generator.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    history.source_intervention_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape[1:], dtype=dtype),
                                                       gc_intervention_selector.out_shape, gc_intervention_selector.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    history.intervention_params_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape[1:], dtype=dtype),
                                                       random_intervention_generator.out_shape, random_intervention_generator.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    history.perturbation_params_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape[1:], dtype=dtype),
                                                       perturbation_generator.out_shape, perturbation_generator.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    history.system_output_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape[1:], dtype=dtype),
                                                       system_rollout.out_shape, system_rollout.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    history.reached_goal_embedding_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape[1:], dtype=dtype),
                                                       goal_embedding_encoder.out_shape, goal_embedding_encoder.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    history.system_rollout_statistics_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape[1:], dtype=dtype),
                                                       rollout_statistics_encoder.out_shape, rollout_statistics_encoder.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    
    # Run Exploration
    for iteration_idx in range(n_random_batches + n_imgep_batches):

        if iteration_idx < n_random_batches:
            # generate random intervention
            key, subkey = jrandom.split(key)
            interventions_params = random_intervention_generator(subkey)
            if out_sanity_check:
                random_intervention_generator.out_sanity_check(interventions_params)

            # empty arrays to fill history
            target_goals_embeddings = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=shape, dtype=dtype),
                                                         goal_generator.out_shape, goal_generator.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
            source_interventions_ids = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=shape, dtype=dtype),
                                                       gc_intervention_selector.out_shape, gc_intervention_selector.out_dtype, is_leaf=lambda node: isinstance(node, tuple))

        else:
            # sample goal
            key, subkey = jrandom.split(key)
            target_goals_embeddings = goal_generator(subkey, history.reached_goal_embedding_library, history.system_rollout_statistics_library)
            if out_sanity_check:
                goal_generator.out_sanity_check(target_goals_embeddings)

            # goal-conditioned selection of source intervention from history
            key, subkey = jrandom.split(key)
            source_interventions_ids = gc_intervention_selector(subkey, target_goals_embeddings, history.reached_goal_embedding_library, history.system_rollout_statistics_library)
            if out_sanity_check:
                gc_intervention_selector.out_sanity_check(source_interventions_ids)
            interventions_params = jtu.tree_map(lambda x: x[source_interventions_ids], history.intervention_params_library)

            # goal-conditioned optimization of source intervention
            key, subkey = jrandom.split(key)
            interventions_params = gc_intervention_optimizer(subkey, intervention_fn, interventions_params, system_rollout, goal_embedding_encoder, goal_achievement_loss, target_goals_embeddings)
            if out_sanity_check:
                gc_intervention_optimizer.out_sanity_check(interventions_params)

        # generate perturbation
        key, subkey = jrandom.split(key)
        perturbations_params = perturbation_generator(subkey)
        if out_sanity_check:
            perturbation_generator.out_sanity_check(perturbations_params)

        # rollout system
        key, subkey = jrandom.split(key)
        system_outputs = system_rollout(subkey, intervention_fn, interventions_params, perturbation_fn, perturbations_params)
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
        history = history.update_node("target_goal_embedding_library", target_goals_embeddings, merge_concatenate)
        history = history.update_node("source_intervention_library", source_interventions_ids, merge_concatenate)
        history = history.update_node("intervention_params_library", interventions_params, merge_concatenate)
        history = history.update_node("perturbation_params_library", perturbations_params, merge_concatenate)
        history = history.update_node("system_output_library", system_outputs, merge_concatenate)
        history = history.update_node("reached_goal_embedding_library", reached_goals_embeddings, merge_concatenate)
        history = history.update_node("system_rollout_statistics_library", system_rollouts_statistics, merge_concatenate)


    # Save history and modules
    history.save(os.path.join(save_folder, "experiment_history.pickle"), overwrite=True)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "random_intervention_generator.eqx"), random_intervention_generator)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "intervention_fn.eqx"), intervention_fn)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "perturbation_generator.eqx"), perturbation_generator)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "perturbation_fn.eqx"), perturbation_fn)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "system_rollout.eqx"), system_rollout)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "rollout_statistics_encoder.eqx"), rollout_statistics_encoder)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "goal_generator.eqx"), goal_generator)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "gc_intervention_selector.eqx"), gc_intervention_selector)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "gc_intervention_optimizer.eqx"), gc_intervention_optimizer)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "goal_embedding_encoder.eqx"), goal_embedding_encoder)
    eqx.tree_serialise_leaves(os.path.join(save_folder, "goal_achievement_loss.eqx"), goal_achievement_loss)