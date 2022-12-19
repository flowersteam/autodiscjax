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

def run_imgep_evaluation(jax_platform_name: str, seed: int, n_perturbations: int, save_folder: str,
                         experiment_system_output_library: adx.DictTree, experiment_intervention_params_library: adx.DictTree, intervention_fn: eqx.Module,
                         perturbation_generator: eqx.Module, perturbation_fn: eqx.Module,
                         system_rollout: eqx.Module, rollout_statistics_encoder: eqx.Module,
                         out_sanity_check=True, save_modules=False):
    
    # Set platform device
    jax.config.update("jax_platform_name", jax_platform_name)

    # Set random seed
    key = jrandom.PRNGKey(seed)

    # set batch_size = # experiments
    batch_size = experiment_system_output_library.ys.shape[0]

    # Initialize History
    history = adx.DictTree()
    history.perturbation_params_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(batch_size, 0, ) + shape, dtype=dtype),
                                                       perturbation_generator.out_shape, perturbation_generator.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    history.system_output_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(batch_size, 0, ) + shape, dtype=dtype),
                                                       system_rollout.out_shape, system_rollout.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
    history.system_rollout_statistics_library = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=(batch_size, 0, ) + shape, dtype=dtype),
                                                       rollout_statistics_encoder.out_shape, rollout_statistics_encoder.out_dtype, is_leaf=lambda node: isinstance(node, tuple))

    # Vmap modules
    batched_perturbation_generator = vmap(perturbation_generator, in_axes=(0, 0), out_axes=0)
    batched_system_rollout = vmap(system_rollout, in_axes=(0, None, 0, None, 0), out_axes=0)
    batched_rollout_statistics_encoder = vmap(rollout_statistics_encoder, in_axes=(0, 0), out_axes=0)

    # Run Evaluation
    for perturbation_idx in range(n_perturbations):

        # generate perturbation
        print("Generate the perturbation")
        key, *subkeys = jrandom.split(key, num=batch_size+1)
        perturbations_params, log_data = batched_perturbation_generator(jnp.array(subkeys), experiment_system_output_library.ys)
        append_to_log(log_data)
        if out_sanity_check:
            vmap(perturbation_generator.out_sanity_check)(perturbations_params)

        # rollout system
        print("Rollout the system")
        key, *subkeys = jrandom.split(key, num=batch_size+1)
        system_outputs, log_data = batched_system_rollout(jnp.array(subkeys), intervention_fn, experiment_intervention_params_library, perturbation_fn, perturbations_params)
        append_to_log(log_data)
        if out_sanity_check:
            vmap(system_rollout.out_sanity_check)(system_outputs)

        # represent outputs -> other statistics
        print("Encode the rollout statistics")
        key, *subkeys = jrandom.split(key, num=batch_size+1)
        system_rollouts_statistics, log_data = batched_rollout_statistics_encoder(jnp.array(subkeys), system_outputs)
        append_to_log(log_data)
        if out_sanity_check:
            vmap(rollout_statistics_encoder.out_sanity_check)(system_rollouts_statistics)


        # Append to history
        perturbations_params = jtu.tree_map(lambda val: val[:, jnp.newaxis], perturbations_params)
        system_outputs = jtu.tree_map(lambda val: val[:, jnp.newaxis], system_outputs)
        system_rollouts_statistics = jtu.tree_map(lambda val: val[:, jnp.newaxis], system_rollouts_statistics)
        history = history.update_node("perturbation_params_library", perturbations_params, merge_concatenate, axis=1)
        history = history.update_node("system_output_library", system_outputs, merge_concatenate, axis=1)
        history = history.update_node("system_rollout_statistics_library", system_rollouts_statistics, merge_concatenate, axis=1)


    # Save history and modules
    history.save(os.path.join(save_folder, "evaluation_history.pickle"), overwrite=True)
    if save_modules:
        eqx.tree_serialise_leaves(os.path.join(save_folder, "perturbation_generator.eqx"), perturbation_generator)
        eqx.tree_serialise_leaves(os.path.join(save_folder, "perturbation_fn.eqx"), perturbation_fn)