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

def run_rs_experiment(jax_platform_name: str, seed: int, n_random_batches: int,
                         batch_size: int, save_folder: str,
                         random_intervention_generator: eqx.Module, intervention_fn: eqx.Module,
                         perturbation_generator: eqx.Module, perturbation_fn: eqx.Module,
                         system_rollout: eqx.Module, rollout_statistics_encoder: eqx.Module,
                         out_sanity_check=True, save_modules=False, logger=None):
    # Set platform device
    jax.config.update("jax_platform_name", jax_platform_name)

    # Set random seed
    key = jrandom.PRNGKey(seed)

    # Initialize History
    history = adx.DictTree()
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
    history.system_rollout_statistics_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(0,) + shape, dtype=dtype),
                                                             rollout_statistics_encoder.out_shape,
                                                             rollout_statistics_encoder.out_dtype,
                                                             is_leaf=lambda node: isinstance(node, tuple))

    # Vmap modules
    batched_random_intervention_generator = vmap(random_intervention_generator, in_axes=(0,), out_axes=(0, None))
    batched_perturbation_generator = vmap(perturbation_generator, in_axes=(0,), out_axes=(0, None))
    batched_system_rollout = vmap(system_rollout, in_axes=(0, None, 0, None, 0), out_axes=(0, None))
    batched_rollout_statistics_encoder = vmap(rollout_statistics_encoder, in_axes=(0, 0), out_axes=(0, None))

    # Run Exploration

    tstart = time.time()
    for iteration_idx in range(n_random_batches):

        print("Generate random intervention")
        # generate random intervention
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        interventions_params, log_data = batched_random_intervention_generator(jnp.array(subkeys))
        append_to_log(log_data)
        if out_sanity_check:
            vmap(random_intervention_generator.out_sanity_check)(interventions_params)

        # generate perturbation
        print("Generate the perturbation")
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        perturbations_params, log_data = batched_perturbation_generator(jnp.array(subkeys))
        if out_sanity_check:
            vmap(perturbation_generator.out_sanity_check)(perturbations_params)

        # rollout system
        print("Rollout the system")
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        system_outputs, log_data = batched_system_rollout(jnp.array(subkeys), intervention_fn, interventions_params,
                                                perturbation_fn, perturbations_params)
        append_to_log(log_data)
        if out_sanity_check:
            vmap(system_rollout.out_sanity_check)(system_outputs)


        # represent outputs -> other statistics
        print("Encode the rollout statistics")
        key, *subkeys = jrandom.split(key, num=batch_size + 1)
        system_rollouts_statistics, log_data = batched_rollout_statistics_encoder(jnp.array(subkeys), system_outputs)
        append_to_log(log_data)
        if out_sanity_check:
            vmap(rollout_statistics_encoder.out_sanity_check)(system_rollouts_statistics)

        # Append to history
        history = history.update_node("intervention_params_library", interventions_params, merge_concatenate)
        history = history.update_node("perturbation_params_library", perturbations_params, merge_concatenate)
        history = history.update_node("system_output_library", system_outputs, merge_concatenate)
        history = history.update_node("system_rollout_statistics_library", system_rollouts_statistics, merge_concatenate)

        # Save history and modules
        history.save(os.path.join(save_folder, "experiment_history.pickle"), overwrite=True)
        if save_modules:
            eqx.tree_serialise_leaves(os.path.join(save_folder, "random_intervention_generator.eqx"), random_intervention_generator)
            eqx.tree_serialise_leaves(os.path.join(save_folder, "intervention_fn.eqx"), intervention_fn)
            eqx.tree_serialise_leaves(os.path.join(save_folder, "perturbation_generator.eqx"), perturbation_generator)
            eqx.tree_serialise_leaves(os.path.join(save_folder, "perturbation_fn.eqx"), perturbation_fn)
            eqx.tree_serialise_leaves(os.path.join(save_folder, "system_rollout.eqx"), system_rollout)
            eqx.tree_serialise_leaves(os.path.join(save_folder, "rollout_statistics_encoder.eqx"), rollout_statistics_encoder)

        if logger is not None:
            logger.save()

    tend = time.time()
    print(tend - tstart)
    if logger is not None:
        logger.add_value("experiment_time", tend - tstart)
        logger.save()
