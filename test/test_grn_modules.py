from autodiscjax import DictTree
import autodiscjax.modules.grnwrappers as grn
from autodiscjax.utils.misc import wall_sticky_collision, wall_elastic_collision, wall_force_field_collision
import importlib
from jax import vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import sbmltoodejax
from tempfile import NamedTemporaryFile


def test_wall_intervention():
    key = jrandom.PRNGKey(0)

    # Load the system
    biomodel_id = 341
    biomodel_odejax_file = NamedTemporaryFile(suffix=".py")
    biomodel_xml_body = sbmltoodejax.biomodels_api.get_content_for_model(biomodel_id)
    model_data = sbmltoodejax.parse.ParseSBMLFile(biomodel_xml_body)
    sbmltoodejax.modulegeneration.GenerateModel(model_data, biomodel_odejax_file.name)
    spec = importlib.util.spec_from_file_location("ModelSpec", biomodel_odejax_file.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grnstep_cls = getattr(module, "ModelStep")
    grnstep = grnstep_cls()
    y0 = getattr(module, "y0")
    w0 = getattr(module, "w0")
    c = getattr(module, "c")
    t0 = getattr(module, "t0")
    system_rollout = grn.GRNRollout(n_steps=100, y0=y0, w0=w0, c=c, t0=t0,
                                    deltaT=0.1, grn_step=grnstep)

    # Wall perturbation Module
    n_walls = 3
    perturbed_node_ids = [0, 1]
    perturbed_intervals = [[0, 100]]
    walls_target_intersection_window = jnp.r_[20:80]
    walls_length_range = [0.1, 0.1]
    walls_sigma = jnp.array([1e-2, 1e-4])

    perturbation_fn = grn.PiecewiseWallCollisionIntervention(
        time_to_interval_fn=grn.TimeToInterval(intervals=perturbed_intervals),
        #collision_fn=jtu.Partial(vmap(wall_elastic_collision, in_axes=(None, None, 0, 0), out_axes=(0, 0))))
        collision_fn=jtu.Partial(vmap(jtu.Partial(wall_force_field_collision), in_axes=(None, None, 0, 0), out_axes=(0, 0))))

    perturbation_params_tree = DictTree()
    for y_idx in perturbed_node_ids:
        perturbation_params_tree.y[y_idx] = "placeholder"
    perturbation_params_tree.sigma = "placeholder"

    perturbation_params_treedef = jtu.tree_structure(perturbation_params_tree)

    perturbation_params_shape = jtu.tree_map(lambda _: (n_walls, 2, len(perturbed_intervals),),
                                             perturbation_params_tree)
    perturbation_params_dtype = jtu.tree_map(lambda _: jnp.float32, perturbation_params_tree)

    perturbation_generator = grn.WallPerturbationGenerator(perturbation_params_treedef,
                                                           perturbation_params_shape,
                                                           perturbation_params_dtype,
                                                           walls_target_intersection_window,
                                                           walls_length_range,
                                                           walls_sigma)

    # Batch modules
    batched_perturbation_generator = vmap(perturbation_generator, in_axes=(0, 0), out_axes=0)
    batched_system_rollout = vmap(system_rollout, in_axes=(0, None, 0, None, 0), out_axes=0)

    # Rollouts with default trajectory
    batch_size = 3
    key, *subkeys = jrandom.split(key, num=batch_size+1)
    default_system_outputs, log_data = batched_system_rollout(jnp.array(subkeys), None, None, None, None)
    ys = default_system_outputs.ys

    # Sample wall perturbation params
    key, *subkeys = jrandom.split(key, num=batch_size+1)
    perturbation_params, log_data = batched_perturbation_generator(jnp.array(subkeys), ys)

    # Evaluation Rollout with wall perturbations
    key, *subkeys = jrandom.split(key, num=batch_size+1)
    wall_system_outputs, log_data = batched_system_rollout(jnp.array(subkeys), None, None, perturbation_fn, perturbation_params)

    # Show results
    for sample_idx in range(batch_size):
        plt.figure()
        plt.plot(ys[sample_idx, perturbed_node_ids[0]], ys[sample_idx, perturbed_node_ids[1]])
        plt.plot(wall_system_outputs.ys[sample_idx, perturbed_node_ids[0]], wall_system_outputs.ys[sample_idx, perturbed_node_ids[1]])
        for wall_idx in range(n_walls):
            plt.plot(perturbation_params.y[perturbed_node_ids[0]][sample_idx, wall_idx].squeeze(),
                     perturbation_params.y[perturbed_node_ids[1]][sample_idx, wall_idx].squeeze())
        plt.show()

