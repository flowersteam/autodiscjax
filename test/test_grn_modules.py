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

    # Default Rollout
    key, subkey = jrandom.split(key)
    default_system_outputs, log_data = system_rollout(subkey)
    ys = default_system_outputs.ys


    # Prepare Wall Intervention
    wall_target_node_idx = 0
    wall_other_node_idx = 1
    time_intervals = [[0, 10]]
    n_walls = 1
    target_intersection_idx = ys.shape[-1] // 2
    wall_target_pos = ys[wall_target_node_idx, target_intersection_idx]
    wall_other_pos = ys[wall_other_node_idx, target_intersection_idx]
    wall_length = 0.1*(ys[wall_other_node_idx].max() - ys[wall_other_node_idx].min())
    intervention_params = DictTree()
    intervention_params.y[wall_target_node_idx] = wall_target_pos * jnp.ones((n_walls, 2, len(time_intervals)))
    intervention_params.y[wall_other_node_idx] = jnp.array([wall_other_pos-wall_length/2., wall_other_pos+wall_length/2.]).reshape((n_walls, 2, len(time_intervals))) * jnp.ones((n_walls, 2, len(time_intervals)))

    collision_fn = jtu.Partial(vmap(wall_elastic_collision, in_axes=(None, None, 0, 0), out_axes=(0,0)))
    intervention_fn = grn.PiecewiseWallCollisionIntervention(time_to_interval_fn=grn.TimeToInterval(intervals=time_intervals), collision_fn=collision_fn)


    # Rollout with wall
    key, subkey = jrandom.split(key)
    wall_system_outputs, log_data = system_rollout(subkey, intervention_fn, intervention_params)

    # Show results
    plt.figure()
    plt.plot(ys[wall_target_node_idx], ys[wall_other_node_idx])
    plt.plot(wall_system_outputs.ys[wall_target_node_idx], wall_system_outputs.ys[wall_other_node_idx])
    plt.plot(intervention_params.y[wall_target_node_idx].squeeze(), intervention_params.y[wall_other_node_idx].squeeze())
    plt.show()

def test_wall_intervention_batch_mode():
    key = jrandom.PRNGKey(0)
    batch_size = 5

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
    key, subkey = jrandom.split(key)
    y0 = jrandom.uniform(subkey, shape=(batch_size, )+y0.shape, minval=0., maxval=300.)
    w0 = jnp.tile(w0, (batch_size, 1))
    system_rollout = grn.GRNRollout(n_steps=100, y0=y0, w0=w0, c=c, t0=t0,
                                    deltaT=0.1, grn_step=grnstep)

    # Default Rollout
    key, subkey = jrandom.split(key)
    default_system_outputs, log_data = system_rollout(subkey)
    ys = default_system_outputs.ys

    # Prepare Wall Intervention
    wall_target_node_idx = 0
    wall_other_node_idx = 1
    time_intervals = [[0, 10]]
    n_walls = 2
    target_intersection_ids = [ys.shape[-1] // 4, 3 * ys.shape[-1] // 4]
    wall_target_pos = ys[:, wall_target_node_idx, target_intersection_ids]
    wall_other_pos = ys[:, wall_other_node_idx, target_intersection_ids]
    wall_length = 0.1 * (ys[:, wall_other_node_idx].max(-1) - ys[:, wall_other_node_idx].min(-1))
    intervention_params = DictTree()
    intervention_params.y[wall_target_node_idx] = jnp.repeat(wall_target_pos[..., jnp.newaxis, jnp.newaxis], 2, 2)
    intervention_params.y[wall_other_node_idx] = jnp.stack(
        [wall_other_pos - wall_length[:, jnp.newaxis] / 2., wall_other_pos + wall_length[:, jnp.newaxis] / 2.],
        axis=-1).reshape(
        (batch_size, n_walls, 2, len(time_intervals))) * jnp.ones((batch_size, n_walls, 2, len(time_intervals)))

    collision_fn = jtu.Partial(
        vmap(vmap(wall_elastic_collision, in_axes=(None, None, 0, 0), out_axes=(0, 0)), in_axes=(0, 0, 0, 0),
             out_axes=(0, 0)))
    intervention_fn = grn.PiecewiseWallCollisionIntervention(
        time_to_interval_fn=grn.TimeToInterval(intervals=time_intervals), collision_fn=collision_fn)

    # Rollout with wall
    key, subkey = jrandom.split(key)
    wall_system_outputs, log_data = system_rollout(subkey, intervention_fn, intervention_params)

    # Show results
    for batch_idx in range(batch_size):
        plt.figure()
        plt.plot(ys[batch_idx, wall_target_node_idx], ys[batch_idx, wall_other_node_idx])
        plt.plot(wall_system_outputs.ys[batch_idx, wall_target_node_idx],
                 wall_system_outputs.ys[batch_idx, wall_other_node_idx])
        plt.plot(intervention_params.y[wall_target_node_idx][batch_idx, 0].squeeze(),
                 intervention_params.y[wall_other_node_idx][batch_idx, 0].squeeze())
        plt.plot(intervention_params.y[wall_target_node_idx][batch_idx, 1].squeeze(),
                 intervention_params.y[wall_other_node_idx][batch_idx, 1].squeeze())
        plt.show()

