import autodiscjax as adx
import equinox as eqx
from autodiscjax.utils.accessors import merge_concatenate
from autodiscjax.utils.logging import append_to_log
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import os
import time


def extract_workers_data_from_optimizer_log(log_data, target_goals_embeddings, source_interventions_ids, history, append_to_history=True, logger=None, append_modules_log_data=True):
    n_optim_steps = len(log_data)

    for optim_step_idx in range(n_optim_steps):
        step_log_data = log_data[optim_step_idx]
        batch_size, n_workers = step_log_data.goal_embedding_encoder.outputs.shape[:2]

        # Reshape outputs to  batch_size*n_workers, ....
        repeated_target_goals_embeddings = jtu.tree_map(lambda node: jnp.repeat(node, n_workers, axis=0), target_goals_embeddings)
        if optim_step_idx == 0:
            source_interventions_ids = jtu.tree_map(lambda node: jnp.repeat(node, n_workers, axis=0), source_interventions_ids)
        else:
            rel_interventions_ids = jnp.repeat(jnp.arange(batch_size)*n_workers, n_workers, axis=0) + step_log_data.source_workers_ids.reshape((batch_size*n_workers, ))
            source_interventions_ids = jnp.arange(len(history.source_intervention_library)-batch_size*n_workers, len(history.source_intervention_library))[rel_interventions_ids]

        interventions_params = jtu.tree_map(lambda node: node.reshape((batch_size*n_workers, ) + node.shape[2:]), step_log_data.workers_params)
        perturbations_params = jtu.tree_map(lambda node: node.reshape((batch_size*n_workers, ) + node.shape[2:]), step_log_data.perturbation_generator.outputs)
        system_outputs = jtu.tree_map(lambda node: node.reshape((batch_size*n_workers, ) + node.shape[2:]), step_log_data.system_rollout.outputs)
        reached_goals_embeddings = jtu.tree_map(lambda node: node.reshape((batch_size*n_workers, ) + node.shape[2:]), step_log_data.goal_embedding_encoder.outputs)
        gc_losses = jtu.tree_map(lambda node: node.reshape((batch_size*n_workers, ) + node.shape[2:]), step_log_data.goal_achievement_loss.outputs)
        system_rollouts_statistics = jtu.tree_map(lambda node: node.reshape((batch_size*n_workers, ) + node.shape[2:]), step_log_data.rollout_statistics_encoder.outputs)

        # Save to history
        if append_to_history:
            history = history.update_node("target_goal_embedding_library", repeated_target_goals_embeddings, merge_concatenate)
            history = history.update_node("source_intervention_library", source_interventions_ids, merge_concatenate)
            history = history.update_node("intervention_params_library", interventions_params, merge_concatenate)
            history = history.update_node("perturbation_params_library", perturbations_params, merge_concatenate)
            history = history.update_node("system_output_library", system_outputs, merge_concatenate)
            history = history.update_node("reached_goal_embedding_library", reached_goals_embeddings, merge_concatenate)
            history = history.update_node("gc_loss_library", gc_losses, merge_concatenate)
            history = history.update_node("system_rollout_statistics_library", system_rollouts_statistics, merge_concatenate)

        # Save logs
        if logger is not None and append_modules_log_data:
            append_to_log(step_log_data.perturbation_generator.log_data)
            append_to_log(step_log_data.system_rollout.log_data)
            append_to_log(step_log_data.goal_embedding_encoder.log_data)
            append_to_log(step_log_data.goal_achievement_loss.log_data)
            append_to_log(step_log_data.rollout_statistics_encoder.log_data)

    del log_data

    return history

def run_imgep_experiment(jax_platform_name: str, seed: int, n_random_batches: int, n_imgep_batches: int,
                         batch_size: int, save_folder: str,
                         random_intervention_generator: eqx.Module, intervention_fn: eqx.Module,
                         perturbation_generator: eqx.Module, perturbation_fn: eqx.Module,
                         system_rollout: eqx.Module, rollout_statistics_encoder: eqx.Module,
                         goal_generator: eqx.Module, gc_intervention_selector: eqx.Module,
                         gc_intervention_optimizer: eqx.Module,
                         goal_embedding_encoder: eqx.Module,
                         goal_achievement_loss: eqx.Module,
                         out_sanity_check=True, save_modules=False, logger=None):
    # Set platform device
    jax.config.update("jax_platform_name", jax_platform_name)

    # Set random seed
    key = jrandom.PRNGKey(seed)

    if out_sanity_check:
        assert (goal_generator.out_treedef == goal_embedding_encoder.out_treedef) \
               and (goal_generator.out_shape == goal_embedding_encoder.out_shape) \
               and (goal_generator.out_dtype == goal_embedding_encoder.out_dtype), \
            "goal generator and goal encoder must operate in same spaces"

        assert (random_intervention_generator.out_treedef == gc_intervention_optimizer.out_treedef) \
               and (random_intervention_generator.out_shape == gc_intervention_optimizer.out_shape) \
               and (random_intervention_generator.out_dtype == gc_intervention_optimizer.out_dtype), \
            "random intervention generator and goal-conditionned intervention operator must operate in same spaces"

    # Initialize History
    history = adx.DictTree()
    history.target_goal_embedding_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape, dtype=dtype),
                                                         goal_generator.out_shape, goal_generator.out_dtype,
                                                         is_leaf=lambda node: isinstance(node, tuple))
    history.source_intervention_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape, dtype=dtype),
                                                       gc_intervention_selector.out_shape,
                                                       gc_intervention_selector.out_dtype,
                                                       is_leaf=lambda node: isinstance(node, tuple))
    history.intervention_params_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape, dtype=dtype),
                                                       random_intervention_generator.out_shape,
                                                       random_intervention_generator.out_dtype,
                                                       is_leaf=lambda node: isinstance(node, tuple))
    history.perturbation_params_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape, dtype=dtype),
                                                       perturbation_generator.out_shape,
                                                       perturbation_generator.out_dtype,
                                                       is_leaf=lambda node: isinstance(node, tuple))
    history.system_output_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape, dtype=dtype),
                                                 system_rollout.out_shape, system_rollout.out_dtype,
                                                 is_leaf=lambda node: isinstance(node, tuple))
    history.reached_goal_embedding_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape, dtype=dtype),
                                                          goal_embedding_encoder.out_shape,
                                                          goal_embedding_encoder.out_dtype,
                                                          is_leaf=lambda node: isinstance(node, tuple))
    history.gc_loss_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape, dtype=dtype),
                                                          goal_achievement_loss.out_shape,
                                                          goal_achievement_loss.out_dtype,
                                                          is_leaf=lambda node: isinstance(node, tuple))
    history.system_rollout_statistics_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape, dtype=dtype),
                                                             rollout_statistics_encoder.out_shape,
                                                             rollout_statistics_encoder.out_dtype,
                                                             is_leaf=lambda node: isinstance(node, tuple))

    # Prepare optimizer
    partial_gc_intervention_optimizer = jtu.Partial(gc_intervention_optimizer,
                                            perturbation_generator=perturbation_generator, perturbation_fn=perturbation_fn,
                                            intervention_fn=intervention_fn, system_rollout=system_rollout,
                                            goal_embedding_encoder=goal_embedding_encoder, goal_achievement_loss=goal_achievement_loss,
                                            rollout_statistics_encoder=rollout_statistics_encoder
                                            )

    # Vmap modules
    batched_random_intervention_generator = vmap(random_intervention_generator, in_axes=(0,), out_axes=(0, None))
    batched_perturbation_generator = vmap(perturbation_generator, in_axes=(0,), out_axes=(0, None))
    batched_system_rollout = vmap(system_rollout, in_axes=(0, None, 0, None, 0), out_axes=(0, None))
    batched_rollout_statistics_encoder = vmap(rollout_statistics_encoder, in_axes=(0, 0), out_axes=(0, None))
    batched_goal_generator = vmap(goal_generator, in_axes=(0, None, None, None), out_axes=(0, None))
    batched_gc_intervention_selector = vmap(gc_intervention_selector, in_axes=(0, 0, None, None), out_axes=(0, None))
    batched_gc_intervention_optimizer = vmap(partial_gc_intervention_optimizer, in_axes=(0, 0, 0), out_axes=(0, 0))
    batched_goal_embedding_encoder = vmap(goal_embedding_encoder, in_axes=(0, 0), out_axes=(0, None))
    batched_goal_achievement_loss = vmap(goal_achievement_loss, in_axes=(0, 0, 0), out_axes=(0, None))

    # Random rollouts
    tstart = time.time()
    for iteration_idx in range(n_random_batches):
        print("Generate random intervention")
        # generate random intervention
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        interventions_params, log_data = batched_random_intervention_generator(jnp.array(subkeys))
        if logger is not None:
            append_to_log(log_data)
        if out_sanity_check:
            vmap(random_intervention_generator.out_sanity_check)(interventions_params)

        # empty arrays to fill history
        target_goals_embeddings = jtu.tree_map(
            lambda shape, dtype: jnp.zeros(shape=(batch_size,) + shape, dtype=dtype),
            goal_generator.out_shape, goal_generator.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
        source_interventions_ids = jtu.tree_map(
            lambda shape, dtype: jnp.zeros(shape=(batch_size,) + shape, dtype=dtype),
            gc_intervention_selector.out_shape, gc_intervention_selector.out_dtype,
            is_leaf=lambda node: isinstance(node, tuple))

        # generate perturbation
        print("Generate the perturbation")
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        perturbations_params, log_data = batched_perturbation_generator(jnp.array(subkeys))
        if logger is not None:
            append_to_log(log_data)
        if out_sanity_check:
            vmap(perturbation_generator.out_sanity_check)(perturbations_params)

        # rollout system
        print("Rollout the system")
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        system_outputs, log_data = batched_system_rollout(jnp.array(subkeys), intervention_fn, interventions_params,
                                                          perturbation_fn, perturbations_params)
        if logger is not None:
            append_to_log(log_data)
        if out_sanity_check:
            vmap(system_rollout.out_sanity_check)(system_outputs)

        # represent outputs -> reached goals
        print("Encode the reached goal")
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        reached_goals_embeddings, log_data = batched_goal_embedding_encoder(jnp.array(subkeys), system_outputs)
        if logger is not None:
            append_to_log(log_data)
        if out_sanity_check:
            vmap(goal_embedding_encoder.out_sanity_check)(reached_goals_embeddings)

        # Compute reached goals -> target goal-conditionned loss
        print("Compute the distance to target goal")
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        gc_losses, log_data = batched_goal_achievement_loss(jnp.array(subkeys), reached_goals_embeddings,
                                                            target_goals_embeddings)
        if logger is not None:
            append_to_log(log_data)
        if out_sanity_check:
            vmap(goal_achievement_loss.out_sanity_check)(gc_losses)

        # represent outputs -> other statistics
        print("Encode the rollout statistics")
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        system_rollouts_statistics, log_data = batched_rollout_statistics_encoder(jnp.array(subkeys),
                                                                                  system_outputs)
        if logger is not None:
            append_to_log(log_data)
        if out_sanity_check:
            vmap(rollout_statistics_encoder.out_sanity_check)(system_rollouts_statistics)

        # Append to history
        print("Append to history")
        history = history.update_node("target_goal_embedding_library", target_goals_embeddings, merge_concatenate)
        history = history.update_node("source_intervention_library", source_interventions_ids, merge_concatenate)
        history = history.update_node("intervention_params_library", interventions_params, merge_concatenate)
        history = history.update_node("perturbation_params_library", perturbations_params, merge_concatenate)
        history = history.update_node("system_output_library", system_outputs, merge_concatenate)
        history = history.update_node("reached_goal_embedding_library", reached_goals_embeddings, merge_concatenate)
        history = history.update_node("gc_loss_library", gc_losses, merge_concatenate)
        history = history.update_node("system_rollout_statistics_library", system_rollouts_statistics,
                                      merge_concatenate)


    # IMGEP rollouts
    for iteration_idx in range(n_imgep_batches):

        # sample goal
        print("Generate target goals")
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        target_goals_embeddings, log_data = batched_goal_generator(jnp.array(subkeys), history.target_goal_embedding_library,
                                                         history.reached_goal_embedding_library,
                                                         history.system_rollout_statistics_library)
        if logger is not None:
            append_to_log(log_data)
        if out_sanity_check:
            vmap(goal_generator.out_sanity_check)(target_goals_embeddings)

        # goal-conditioned selection of source intervention from history
        print("Select closes intervention")
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        source_interventions_ids, log_data = batched_gc_intervention_selector(jnp.array(subkeys), target_goals_embeddings,
                                                                    history.reached_goal_embedding_library,
                                                                    history.system_rollout_statistics_library)
        if logger is not None:
            append_to_log(log_data)
        if out_sanity_check:
            vmap(gc_intervention_selector.out_sanity_check)(source_interventions_ids)
        interventions_params = jtu.tree_map(lambda x: x[source_interventions_ids],
                                            history.intervention_params_library)

        # goal-conditioned optimization of source intervention
        print("Optimize the selected intervention")
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        interventions_params, log_data = batched_gc_intervention_optimizer(jnp.array(subkeys), interventions_params, target_goals_embeddings)
        if out_sanity_check:
            vmap(gc_intervention_optimizer.out_sanity_check)(interventions_params)

        print("Extract workers data from optimizer logs")
        history = extract_workers_data_from_optimizer_log(log_data, target_goals_embeddings, source_interventions_ids, history, append_to_history=True, logger=logger, append_modules_log_data=True)

    # Save history and modules
    print("Save history")
    history.save(os.path.join(save_folder, "experiment_history.pickle"), overwrite=True)
    if save_modules:
        print("Save modules")
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

    tend = time.time()
    if logger is not None:
        print("Save logger")
        logger.add_value("experiment_time", tend - tstart)
        logger.save()
    print(f"Total time : {tend - tstart}")

