import autodiscjax as adx
from autodiscjax.modules.imgepwrappers import BaseSystemRollout, BaseRolloutStatisticsEncoder
from autodiscjax.utils.misc import filter_update, hardplus, normal
from autodiscjax.utils import timeseries
import equinox as eqx
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import PyTree, Array
import time
from typing import Callable, Sequence


class BasePerturbationGenerator(adx.Module):
    @jit
    def __call__(self, key, system_outputs_library):
        raise NotImplementedError

class NoisePerturbationGenerator(adx.Module):
    """
    out_treedef: out_params.y[idx] = Array for idx in perturbed node ids
    out_shape: Array of shape (batch_size, len(time_intervals))
    """
    std: float = 0.1
    normal_fn: Callable

    def __init__(self, out_treedef, out_shape, out_dtype, std):
        super().__init__(out_treedef, out_shape, out_dtype)
        self.std = std
        zero_mean = jtu.tree_map(lambda shape, dtype: jnp.zeros(shape=shape, dtype=dtype), self.out_shape, self.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
        self.normal_fn = jit(jtu.Partial(normal, mean=zero_mean, out_treedef=self.out_treedef, out_shape=self.out_shape, out_dtype=self.out_dtype))


    def __call__(self, key, system_outputs_library):

        std = jtu.tree_map(lambda shape, dtype: self.std*jnp.ones(shape=shape, dtype=dtype), self.out_shape,
                                 self.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
        for y_idx, out_shape in self.out_shape.y.items():
            yranges = (system_outputs_library.ys[..., y_idx, :].max(-1) - system_outputs_library.ys[..., y_idx, :].min(-1))[..., jnp.newaxis] #shape(batch_size, 1)
            std.y[y_idx] = std.y[y_idx] * yranges

        return self.normal_fn(key, std=std)

class WallPerturbationGenerator(adx.Module):
    """
    out_treedef: out_params.y[idx] = Array for idx in [node1, node2] where wall is defined
    out_shape: Array of shape (..., n_walls, 2, len(time_intervals))
    """
    walls_target_intersection_steps: Sequence[int]
    walls_length: Array

    def __init__(self, out_treedef, out_shape, out_dtype, walls_target_intersection_steps, walls_length):
        super().__init__(out_treedef, out_shape, out_dtype)
        assert len(self.out_shape.y.keys()) == 2  # wall is defined on two nodes
        out_shape = jtu.tree_flatten(self.out_shape, is_leaf=lambda node: isinstance(node, tuple))[0][0]
        assert out_shape[1] == len(walls_target_intersection_steps) == len(walls_length)  # n_walls

        self.walls_target_intersection_steps = walls_target_intersection_steps
        self.walls_length = walls_length


    @jit
    def __call__(self, key, ys):
        out_params = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=shape, dtype=dtype), self.out_shape,
                           self.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
        out_shape = jtu.tree_flatten(self.out_shape, is_leaf=lambda node: isinstance(node, tuple))[0][0]

        target_node_idx, other_node_idx = out_params.y.keys()
        walls_target_centers = ys[..., target_node_idx, self.walls_target_intersection_steps]
        walls_other_centers = ys[..., other_node_idx, self.walls_target_intersection_steps]
        walls_length = self.walls_length * (ys[..., other_node_idx, :].max(-1) - ys[..., other_node_idx, :].min(-1))[..., jnp.newaxis]
        out_params.y[target_node_idx] = jnp.repeat(walls_target_centers[..., jnp.newaxis, jnp.newaxis], 2, -2)
        out_params.y[other_node_idx] = jnp.stack([walls_other_centers - walls_length / 2.,
                                                  walls_other_centers + walls_length / 2.], axis=-1).reshape(out_shape) * \
                                       jnp.ones(out_shape)

        return out_params

class NullIntervention(eqx.Module):
    @jit
    def __call__(self, key, y, y_, w, w_, c, c_, t_, intervention_params):
        return y, w, c

class TimeToInterval(eqx.Module):
    intervals: Sequence

    @jit
    def __call__(self, t):
        return jnp.where(jnp.array([jnp.logical_and(t >= interval[0], t <= interval[1]) for interval in self.intervals]), size=1, fill_value=-1)[0][0]


class PiecewiseIntervention(eqx.Module):
    time_to_interval_fn: eqx.Module # function is defined as an equinox Module to be jitable
    null_intervention: NullIntervention

    def __init__(self, time_to_interval_fn):
        """
        Piecewise Intervention
        Parameters:
            - time_to_interval is a module that takes t (float - secs) as input and returns the interval_idx (array of int of size 1) with interval_idx=[-1] for null intervention.
        """
        self.time_to_interval_fn = time_to_interval_fn
        self.null_intervention = NullIntervention()

    @jit
    def __call__(self, key, y, y_, w, w_, c, c_, t_, intervention_params):
        """
        If time_to_interval_fn(t) returns -1, apply NullIntervention else apply intervention on corresponding interval
        """
        interval_idx = self.time_to_interval_fn(t_)
        return lax.cond(interval_idx.sum() >= 0, self.apply, self.null_intervention, key, y, y_, w, w_, c, c_, interval_idx, intervention_params)

    @jit
    def apply(self, key, y, y_, w, w_, c, c_, interval_idx, intervention_params):
        raise NotImplementedError


class PiecewiseSetConstantIntervention(PiecewiseIntervention):
    """
    intervention_params shape must be (batch_size, len(intervals))
    """

    @jit
    def apply(self, key, y, y_, w, w_, c, c_, interval_idx, intervention_params):
        for y_idx, new_val in intervention_params.y.items():
            y = y.at[..., y_idx].set(hardplus(new_val[..., interval_idx]))
        for w_idx, new_val in intervention_params.w.items():
            w = w.at[..., w_idx].set(hardplus(new_val[..., interval_idx]))
        for c_idx, new_val in intervention_params.c.items():
            c = c.at[..., c_idx].set(hardplus(new_val[..., interval_idx]))

        return y, w, c

class PiecewiseAddConstantIntervention(PiecewiseIntervention):
    @jit
    def apply(self, key, y, y_, w, w_, c, c_, interval_idx, intervention_params):
        for y_idx, new_val in intervention_params.y.items():
            y = y.at[..., y_idx].add(new_val[..., interval_idx])
        for w_idx, new_val in intervention_params.w.items():
            w = w.at[..., w_idx].add(new_val[..., interval_idx])
        for c_idx, new_val in intervention_params.c.items():
            c = c.at[..., c_idx].add(new_val[..., interval_idx])

        y = jnp.maximum(y, 0.0)
        return y, w, c

class PiecewiseWallCollisionIntervention(PiecewiseIntervention):
    collision_fn: Callable

    def __init__(self, time_to_interval_fn, collision_fn):
        super().__init__(time_to_interval_fn)
        self.collision_fn = collision_fn

    @jit
    def apply(self, key, y, y_, w, w_, c, c_, interval_idx, intervention_params):
        """
        The walls are described by
          - a start point w_: intervention_params.y[*][:, wall_idx, 0, interval_idx]
          - a end point w: intervention_params.y[*][:, wall_idx, 1, interval_idx]
        The grn trajectory is described by its start point: y_[*] and its end point: y[*]
        If there is an intersection between the grn trajectory and the wall, we replace y by the intersection point.
        If they are several walls, the closest intersection point is selected.
        """

        wall_dim = len(intervention_params.y.keys())
        if wall_dim != 2:
            raise NotImplementedError
        wall_params_shape = jtu.tree_leaves(intervention_params)[0].shape #..., n_walls, 2, len(intervals)
        n_walls = wall_params_shape[-3]
        assert wall_params_shape[-2] == 2 #start and end point
        walls = jnp.empty(wall_params_shape[:-3]+(n_walls, 2, wall_dim))
        grn_traj = jnp.empty(wall_params_shape[:-3]+(2, wall_dim))
        for rel_idx, (y_idx, wall_val) in enumerate(intervention_params.y.items()):
            walls = walls.at[..., rel_idx].set(wall_val[..., interval_idx])
            grn_traj = grn_traj.at[..., 0, rel_idx].set(y_[..., y_idx])
            grn_traj = grn_traj.at[..., 1, rel_idx].set(y[..., y_idx])

        t_collisions, p_after_collisions = self.collision_fn(grn_traj[..., 0, :], grn_traj[..., 1, :], walls[..., 0, :], walls[..., 1, :])
        closest_intersection_idx = jnp.nanargmin(t_collisions, axis=-1)
        closest_intersection_idx = closest_intersection_idx.reshape(closest_intersection_idx.shape + (1, 1))
        new_y = jnp.take_along_axis(p_after_collisions, closest_intersection_idx, axis=-2).squeeze()
        y = y.at[..., list(intervention_params.y.keys())].set(new_y)

        if "w" in intervention_params.keys() or "c" in intervention_params.keys():
            raise NotImplementedError

        return y, w, c

class GRNRollout(BaseSystemRollout):
    n_steps: int = eqx.static_field()
    deltaT: float = eqx.static_field()
    y0: Array
    w0: Array
    c: Array
    t0: float = eqx.static_field()
    grn_step: eqx.Module
    out_treedef: jtu.PyTreeDef = eqx.static_field()
    out_shape: PyTree = eqx.static_field()
    out_dtype: PyTree = eqx.static_field()

    def __init__(self, n_steps, y0, w0, c, t0, deltaT, grn_step):
        self.n_steps = n_steps
        self.y0 = jnp.maximum(y0, 0.0)
        self.w0 = w0
        self.c = c
        self.t0 = t0
        self.deltaT = deltaT

        vmap_axes = (0 if y0.ndim == 2 else None, 0 if w0.ndim == 2 else None, 0 if c.ndim == 2 else None, None, None)
        if 0 in vmap_axes:
            self.grn_step = vmap(grn_step, in_axes=vmap_axes, out_axes=vmap_axes[:-1])
        else:
            self.grn_step = grn_step

        out_shape = adx.DictTree()
        out_shape.ys = self.y0.shape + (n_steps, )
        out_shape.ws = self.w0.shape + (n_steps, )
        out_shape.cs = self.c.shape + (n_steps,)
        out_shape.ts = (n_steps, )

        out_treedef = jtu.tree_structure(out_shape, is_leaf=lambda node: isinstance(node, tuple))

        out_dtype = jtu.tree_map(lambda _: jnp.float32, out_shape, is_leaf=lambda node: isinstance(node, tuple))

        super().__init__(out_treedef, out_shape, out_dtype)

    def __call__(self, key,
                 intervention_fn=None, intervention_params=None,
                 perturbation_fn=None, perturbation_params=None):

        if intervention_fn is None:
            intervention_fn = NullIntervention()
        if perturbation_fn is None:
            perturbation_fn = NullIntervention()

        rollout_start = time.time()

        @jit
        def f(carry, x):
            (key, y_, w_, c_, t_) = carry

            # make grn step
            y, w, c, t = self.grn_step(y_, w_, c_, t_, self.deltaT)

            # apply intervention
            key, subkey = jrandom.split(key)
            y, w, c = intervention_fn(key, y, y_, w, w_, c, c_, t_, intervention_params)

            # apply perturbation
            key, subkey = jrandom.split(key)
            y, w, c = perturbation_fn(key, y, y_, w, w_, c, c_, t_, perturbation_params)

            return (key, y, w, c, t), (y, w, c, t)

        (key, y, w, c, t), (ys, ws, cs, ts) = lax.scan(f, (key, self.y0, self.w0, self.c, self.t0), jnp.arange(self.n_steps))
        ys = jnp.moveaxis(ys, 0, -1)
        ws = jnp.moveaxis(ws, 0, -1)
        cs = jnp.moveaxis(cs, 0, -1)

        rollout_end = time.time()

        outputs = adx.DictTree()
        outputs.ys = ys
        outputs.ws = ws
        outputs.cs = cs
        outputs.ts = ts

        return outputs

class GRNRolloutStatisticsEncoder(BaseRolloutStatisticsEncoder):
    filter_fn: Callable
    update_fn: Callable
    
    def __init__(self, y_shape, time_window=jnp.r_[-100:0], is_stable_std_epsilon=1e-2, is_converging_filter_size=50,
                 is_periodic_max_frequency_threshold=40, deltaT=0.1
                 ):
        out_shape = adx.DictTree()
        out_shape.mean_vals = y_shape[:-1]
        out_shape.std_vals = y_shape[:-1]
        out_shape.amplitude_vals = y_shape[:-1]
        out_shape.max_frequency_vals = y_shape[:-1]
        out_shape.diff_signs = y_shape[:-1]
        out_shape.is_stable = y_shape[:-1]
        out_shape.is_converging = y_shape[:-1]
        out_shape.is_monotonous = y_shape[:-1]
        out_shape.is_periodic = y_shape[:-1]
        
        out_treedef = jtu.tree_structure(out_shape, is_leaf=lambda node: isinstance(node, tuple))
        
        out_dtype = out_treedef.unflatten([jnp.float32, jnp.float32, jnp.float32, jnp.float32, jnp.float32, jnp.bool_, jnp.bool_, jnp.bool_, jnp.bool_])
        
        super().__init__(out_treedef=out_treedef, out_shape=out_shape, out_dtype=out_dtype)

        self.filter_fn = jit(jtu.Partial(lambda system_outputs: system_outputs.ys))
        self.update_fn = jit(jtu.Partial(self.calc_statistics, time_window=time_window,
                            is_stable_std_epsilon=is_stable_std_epsilon,
                            is_converging_filter_size=is_converging_filter_size,
                            is_periodic_max_frequency_threshold=is_periodic_max_frequency_threshold, deltaT=deltaT))

    def calc_statistics(self, y, time_window, 
                        is_stable_std_epsilon, is_converging_filter_size, 
                        is_periodic_max_frequency_threshold, deltaT):
        is_stable, mean_vals, std_vals = timeseries.is_stable(y, time_window, is_stable_std_epsilon)
        is_converging = timeseries.is_converging(y, time_window, is_converging_filter_size)
        is_monotonous, diff_signs = timeseries.is_monotonous(y, time_window)
        is_periodic, _, amplitude_vals, max_frequency_vals = timeseries.is_periodic(y, time_window, deltaT, is_periodic_max_frequency_threshold)
        
        
        stats = self.out_treedef.unflatten([mean_vals, std_vals, amplitude_vals, max_frequency_vals, diff_signs, is_stable, is_converging, is_monotonous, is_periodic])
        return stats 

    def __call__(self, key, system_outputs):
        return filter_update(system_outputs, self.filter_fn, self.update_fn, self.out_treedef)