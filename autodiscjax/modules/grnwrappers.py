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

    @eqx.filter_jit
    def __call__(self, key, system_outputs_library):

        std = jtu.tree_map(lambda shape, dtype: self.std*jnp.ones(shape=shape, dtype=dtype), self.out_shape,
                                 self.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
        for y_idx, out_shape in self.out_shape.y.items():
            yranges = (system_outputs_library.ys[..., y_idx, :].max(-1) - system_outputs_library.ys[..., y_idx, :].min(-1))[..., jnp.newaxis] #shape(batch_size, 1)
            std.y[y_idx] = std.y[y_idx] * yranges
        for w_idx, out_shape in self.out_shape.w.items():
            wranges = (system_outputs_library.ws[..., w_idx, :].max(-1) - system_outputs_library.ws[..., w_idx, :].min(-1))[..., jnp.newaxis] #shape(batch_size, 1)
            std.w[w_idx] = std.w[w_idx] * wranges
        for c_idx, out_shape in self.out_shape.c.items():
            std.c[c_idx] = std.c[c_idx] * system_outputs_library.cs[..., c_idx, :].max(-1)

        return self.normal_fn(key, std=std), None


class PushPerturbationGenerator(adx.Module):
    """
    out_treedef: out_params.y[idx] = Array for idx in perturbed node ids
    out_shape: Array of shape (batch_size, len(time_intervals))
    """
    magnitude: float = 0.1

    def __init__(self, out_treedef, out_shape, out_dtype, magnitude):
        super().__init__(out_treedef, out_shape, out_dtype)
        self.magnitude = magnitude

    @eqx.filter_jit
    def __call__(self, key, system_outputs_library):

        magnitude = jtu.tree_map(lambda shape, dtype: self.magnitude*jnp.ones(shape=shape, dtype=dtype), self.out_shape,
                                 self.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
        for y_idx, out_shape in self.out_shape.y.items():
            yranges = (system_outputs_library.ys[..., y_idx, :].max(-1) - system_outputs_library.ys[..., y_idx, :].min(-1))[..., jnp.newaxis] #shape(batch_size, 1)
            magnitude.y[y_idx] = magnitude.y[y_idx] * yranges
        for w_idx, out_shape in self.out_shape.w.items():
            wranges = (system_outputs_library.ws[..., w_idx, :].max(-1) - system_outputs_library.ws[..., w_idx, :].min(-1))[..., jnp.newaxis] #shape(batch_size, 1)
            magnitude.w[w_idx] = magnitude.w[w_idx] * wranges
        for c_idx, out_shape in self.out_shape.c.items():
            magnitude.c[c_idx] = magnitude.c[c_idx] * system_outputs_library.cs[..., c_idx, :].max(-1)

        # sample push = adding (-1, 0, or 1)*magnitude over each perturbation axis
        key = self.out_treedef.unflatten(jrandom.split(key, self.out_treedef.num_leaves))
        out_params = jtu.tree_map(lambda node, subkey: jrandom.choice(subkey, jnp.arange(-1,2, dtype=jnp.float32)) * node, magnitude, key)

        return out_params, None

class WallPerturbationGenerator(adx.Module):
    """
    out_treedef: out_params.y[idx] = Array for idx in [node1, node2] where wall is defined
    out_shape: Array of shape (..., n_walls, 2, len(time_intervals))
    """
    n_walls: int
    intersection_windows: Array
    length_ranges: Array
    sigmas: Array

    def __init__(self, out_treedef, out_shape, out_dtype, n_walls, intersection_windows, length_ranges, sigmas):
        super().__init__(out_treedef, out_shape, out_dtype)
        assert len(self.out_shape.y.keys()) == 2  # wall is defined on two nodes

        out_shape = jtu.tree_flatten(self.out_shape, is_leaf=lambda node: isinstance(node, tuple))[0][0]  # ..., n_walls, 2, len(time_intervals)
        assert out_shape[-3] == n_walls
        assert out_shape[-2] == 2
        assert len(intersection_windows) == n_walls and jnp.array([len(intersection_windows[i]) == 2 for i in range(n_walls)]).all()
        assert len(length_ranges) == n_walls and jnp.array([len(length_ranges[i]) == 2 for i in range(n_walls)]).all()
        assert len(sigmas) == 2 and jnp.array([isinstance(sigmas[i], float) for i in range(2)]).all()

        self.n_walls = n_walls
        self.intersection_windows = intersection_windows
        self.length_ranges = jnp.array(length_ranges)
        self.sigmas = jnp.array(sigmas)

    @eqx.filter_jit
    def __call__(self, key, system_outputs_library):
        out_params = jtu.tree_map(lambda shape, dtype: jnp.empty(shape=shape, dtype=dtype), self.out_shape,
                           self.out_dtype, is_leaf=lambda node: isinstance(node, tuple))
        out_shape = jtu.tree_flatten(self.out_shape, is_leaf=lambda node: isinstance(node, tuple))[0][0] #..., n_walls, 2, len(time_intervals)

        node_ids = jnp.array(list(out_params.y.keys()))
        ys = system_outputs_library.ys[..., node_ids, :]

        # sample random wall orientation (50% vertical, 50% horizontal)

        key, subkey = jrandom.split(key)
        target_node_rel_ids = jrandom.choice(subkey, jnp.arange(2), shape=(self.n_walls, ))
        other_node_rel_ids = 1-target_node_rel_ids

        # sample random wall length
        walls_length = []
        for wall_idx in range(self.n_walls):
            key, subkey = jrandom.split(key)
            walls_length.append(jrandom.uniform(subkey, shape=(1, ), minval=self.length_ranges[wall_idx][0], maxval=self.length_ranges[wall_idx][1]))
        walls_length = jnp.concatenate(walls_length)

        # sample random wall position
        distance_travelled = jnp.cumsum(jnp.sqrt(jnp.sum((jnp.diff(ys, axis=-1)/(ys.max(-1) - ys.min(-1))[:, jnp.newaxis])** 2, axis=-2)), axis=-1) # shape (..., n_system_steps)
        distance_travelled = distance_travelled / distance_travelled.max(-1) # normalize between 0 and 1
        walls_target_intersection_step = []
        for wall_idx in range(self.n_walls):
            intersection_window_step_ids = jnp.where((distance_travelled > self.intersection_windows[wall_idx][0]) &
                                                     (distance_travelled < self.intersection_windows[wall_idx][1]), size=distance_travelled.shape[-1], fill_value=-1)[0]
            key, subkey = jrandom.split(key)
            walls_target_intersection_step.append(jrandom.choice(subkey, intersection_window_step_ids, shape=(1, ),
                                                                 p=jnp.ones_like(intersection_window_step_ids) * (intersection_window_step_ids>=0)))
        walls_target_intersection_step = jnp.concatenate(walls_target_intersection_step)

        walls_target_centers = ys[..., target_node_rel_ids, walls_target_intersection_step]
        walls_other_centers = ys[..., other_node_rel_ids, walls_target_intersection_step]
        walls_length = walls_length * (ys[..., other_node_rel_ids, :].max(-1) - ys[..., other_node_rel_ids, :].min(-1))
        walls_target = jnp.repeat(jnp.repeat(walls_target_centers[..., jnp.newaxis, jnp.newaxis], out_shape[-2], -2), out_shape[-1], -1) # repeat over wall dims and over time intervals
        walls_other = jnp.stack([walls_other_centers - walls_length / 2., walls_other_centers + walls_length / 2.], axis=-1).reshape(out_shape) * jnp.ones(out_shape)
        for k, v in out_params.y.items():
            out_params.y[k] = jnp.where((node_ids[target_node_rel_ids] == k)[..., jnp.newaxis, jnp.newaxis], walls_target, walls_other)

        sigma = self.sigmas * jnp.stack([ys[..., target_node_rel_ids, :].max(-1) - ys[..., target_node_rel_ids, :].min(-1),
                                         ys[..., other_node_rel_ids, :].max(-1) - ys[..., other_node_rel_ids, :].min(-1)],
                                                         axis=-1)
        out_params.sigma = jnp.repeat(sigma[..., jnp.newaxis], out_shape[-1], -1) #repeat over time intervals

        return out_params, None

class NullIntervention(eqx.Module):
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

    def __call__(self, key, y, y_, w, w_, c, c_, t_, intervention_params):
        """
        If time_to_interval_fn(t) returns -1, apply NullIntervention else apply intervention on corresponding interval
        """
        interval_idx = self.time_to_interval_fn(t_)
        return lax.cond(interval_idx.sum() >= 0, self.apply, self.null_intervention, key, y, y_, w, w_, c, c_, interval_idx, intervention_params)

    def apply(self, key, y, y_, w, w_, c, c_, interval_idx, intervention_params):
        raise NotImplementedError


class PiecewiseSetConstantIntervention(PiecewiseIntervention):
    """
    intervention_params shape must be (batch_size, len(intervals))
    """
    def apply(self, key, y, y_, w, w_, c, c_, interval_idx, intervention_params):
        for y_idx, new_val in intervention_params.y.items():
            y = y.at[..., y_idx].set(hardplus(new_val[..., interval_idx]))
        for w_idx, new_val in intervention_params.w.items():
            w = w.at[..., w_idx].set(hardplus(new_val[..., interval_idx]))
        for c_idx, new_val in intervention_params.c.items():
            c = c.at[..., c_idx].set(hardplus(new_val[..., interval_idx]))

        return y, w, c

class PiecewiseAddConstantIntervention(PiecewiseIntervention):
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

    def apply(self, key, y, y_, w, w_, c, c_, interval_idx, intervention_params):
        """
        The walls are described by
          - a start point w_: intervention_params.y[*][:, wall_idx, 0, interval_idx]
          - a end point w: intervention_params.y[*][:, wall_idx, 1, interval_idx]
          - [for force field collision] a sigma value: intervention_params.sigma[*][:, wall_idx, *, interval_idx] where sigma[0] is sigma_perp and sigma[1] is sigma_parallel
        The grn trajectory is described by its start point: y_[*] and its end point: y[*]
        If there is an intersection between the grn trajectory and the wall, we replace y by the intersection point.
        If they are several walls, the closest intersection point is selected.
        """

        wall_dim = len(intervention_params.y.keys())
        if wall_dim != 2:
            raise NotImplementedError
        wall_params_shape = jtu.tree_leaves(intervention_params.y)[0].shape #..., n_walls, 2, len(intervals)
        n_walls = wall_params_shape[-3]
        assert wall_params_shape[-2] == 2 #start and end point
        walls = jnp.empty(wall_params_shape[:-3]+(n_walls, 2, wall_dim))
        grn_traj = jnp.empty(wall_params_shape[:-3]+(2, wall_dim))
        for rel_idx, (y_idx, wall_val) in enumerate(intervention_params.y.items()):
            walls = walls.at[..., rel_idx].set(wall_val[..., interval_idx])
            grn_traj = grn_traj.at[..., 0, rel_idx].set(y_[..., y_idx])
            grn_traj = grn_traj.at[..., 1, rel_idx].set(y[..., y_idx])

        t_collisions, p_after_collisions = self.collision_fn(grn_traj[..., 0, :], grn_traj[..., 1, :], walls[..., 0, :], walls[..., 1, :], sigma=intervention_params.sigma[..., interval_idx])
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
        self.grn_step = grn_step

        out_shape = adx.DictTree()
        out_shape.ys = self.y0.shape + (n_steps, )
        out_shape.ws = self.w0.shape + (n_steps, )
        out_shape.cs = self.c.shape + (n_steps,)
        out_shape.ts = (n_steps, )

        out_treedef = jtu.tree_structure(out_shape, is_leaf=lambda node: isinstance(node, tuple))

        out_dtype = jtu.tree_map(lambda _: jnp.float32, out_shape, is_leaf=lambda node: isinstance(node, tuple))

        super().__init__(out_treedef, out_shape, out_dtype)

    @jit
    def __call__(self, key,
                 intervention_fn=None, intervention_params=None,
                 perturbation_fn=None, perturbation_params=None):

        if intervention_fn is None:
            intervention_fn = NullIntervention()
        if perturbation_fn is None:
            perturbation_fn = NullIntervention()

        #rollout_start = time.time()

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

            return (key, y, w, c, t), (y, w, c, t) #(y,w,c,t) and not (y_,w_c_,t_) because y0=y after first intervention (before set to default)

        (key, y, w, c, t), (ys, ws, cs, ts) = lax.scan(f, (key, self.y0, self.w0, self.c, self.t0), jnp.arange(self.n_steps))
        ys = jnp.moveaxis(ys, 0, -1)
        ws = jnp.moveaxis(ws, 0, -1)
        cs = jnp.moveaxis(cs, 0, -1)

        #rollout_end = time.time()
        #log_data = adx.DictTree(system_rollout_time=rollout_end-rollout_start)

        outputs = adx.DictTree()
        outputs.ys = ys
        outputs.ws = ws
        outputs.cs = cs
        outputs.ts = ts

        return outputs, None

class GRNRolloutStatisticsEncoder(BaseRolloutStatisticsEncoder):
    filter_fn: Callable
    update_fn: Callable
    
    def __init__(self, y_shape, is_stable_time_window=jnp.r_[-1000:0], is_stable_std_epsilon=1e-3,
                 is_converging_time_window=jnp.r_[-1000:0], is_converging_ratio_threshold=0.5, is_monotonous_time_window=jnp.r_[-1000:0],
                 is_periodic_time_window=jnp.r_[-1000:0], is_periodic_max_frequency_threshold=40, is_periodic_deltaT=0.1,
                 ):

        out_shape, out_dtype = adx.DictTree(), adx.DictTree()
        out_shape.mean_vals, out_dtype.mean_vals = y_shape[:-1], jnp.float32
        out_shape.std_vals, out_dtype.std_vals = y_shape[:-1], jnp.float32
        out_shape.amplitude_vals, out_dtype.amplitude_vals = y_shape[:-1], jnp.float32
        out_shape.max_frequency_vals, out_dtype.max_frequency_vals = y_shape[:-1], jnp.float32
        out_shape.diff_signs, out_dtype.diff_signs = y_shape[:-1], jnp.float32
        out_shape.is_valid, out_dtype.is_valid = y_shape[:-1], jnp.bool_
        out_shape.is_stable, out_dtype.is_stable = y_shape[:-1], jnp.bool_
        out_shape.is_converging, out_dtype.is_converging = y_shape[:-1], jnp.bool_
        out_shape.is_monotonous, out_dtype.is_monotonous = y_shape[:-1], jnp.bool_
        out_shape.is_periodic, out_dtype.is_periodic = y_shape[:-1], jnp.bool_
        
        out_treedef = jtu.tree_structure(out_shape, is_leaf=lambda node: isinstance(node, tuple))

        super().__init__(out_treedef=out_treedef, out_shape=out_shape, out_dtype=out_dtype)

        self.filter_fn = jtu.Partial(lambda system_outputs: system_outputs.ys)
        self.update_fn = jtu.Partial(self.calc_statistics, is_stable_time_window=is_stable_time_window, is_stable_std_epsilon=is_stable_std_epsilon,
                                     is_converging_time_window=is_converging_time_window, is_converging_ratio_threshold=is_converging_ratio_threshold,
                                     is_monotonous_time_window=is_monotonous_time_window, is_periodic_time_window=is_periodic_time_window,
                                     is_periodic_max_frequency_threshold=is_periodic_max_frequency_threshold, is_periodic_deltaT=is_periodic_deltaT)

    def calc_statistics(self, y, is_stable_time_window, is_stable_std_epsilon, is_converging_time_window, is_converging_ratio_threshold,
                        is_monotonous_time_window, is_periodic_time_window, is_periodic_max_frequency_threshold, is_periodic_deltaT):

        is_valid = ~(jnp.isnan(y).any(-1))
        is_stable, mean_vals, std_vals = timeseries.is_stable(y, time_window=is_stable_time_window, std_epsilon=is_stable_std_epsilon)
        is_converging = timeseries.is_converging(y, time_window=is_converging_time_window, ratio_threshold=is_converging_ratio_threshold)
        is_monotonous, diff_signs = timeseries.is_monotonous(y, time_window=is_monotonous_time_window)
        is_periodic, _, amplitude_vals, max_frequency_vals = timeseries.is_periodic(y, time_window=is_periodic_time_window, deltaT=is_periodic_deltaT,
                                                                                    max_frequency_threshold=is_periodic_max_frequency_threshold)

        stats = adx.DictTree()
        stats.mean_vals = mean_vals
        stats.std_vals = std_vals
        stats.amplitude_vals = amplitude_vals
        stats.max_frequency_vals = max_frequency_vals
        stats.diff_signs = diff_signs
        stats.is_valid = is_valid
        stats.is_stable = is_stable
        stats.is_converging = is_converging
        stats.is_monotonous = is_monotonous
        stats.is_periodic = is_periodic
        
        return stats

    @eqx.filter_jit
    def __call__(self, key, system_outputs):
        return filter_update(system_outputs, self.filter_fn, self.update_fn, self.out_treedef), None