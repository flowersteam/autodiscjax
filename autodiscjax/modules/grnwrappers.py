from autodiscjax import DictTree
from autodiscjax.modules.imgepwrappers import BaseSystemRollout, BaseRolloutStatisticsEncoder
from autodiscjax.utils.misc import filter_update, hardplus
from autodiscjax.utils import timeseries
import equinox as eqx
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import PyTree, Array
from typing import Callable, Sequence

class NullIntervention(eqx.Module):
    @jit
    def __call__(self, key, y, w, c, t, intervention_params):
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
    def __call__(self, key, y, w, c, t, intervention_params):
        """
        If time_to_interval_fn(t) returns -1, apply NullIntervention else apply intervention on corresponding interval
        """
        interval_idx = self.time_to_interval_fn(t)
        return lax.cond(interval_idx.sum() >= 0, self.apply, self.null_intervention, key, y, w, c, interval_idx, intervention_params)

    @jit
    def apply(self, key, y, w, c, interval_idx, intervention_params):
        raise NotImplementedError


class PiecewiseSetConstantIntervention(PiecewiseIntervention):
    """
    intervention_params shape must be (batch_size, len(intervals))
    """

    @jit
    def apply(self, key, y, w, c, interval_idx, intervention_params):
        for y_idx, new_val in intervention_params.y.items():
            y = y.at[..., y_idx].set(hardplus(new_val[..., interval_idx]))
        for w_idx, new_val in intervention_params.w.items():
            w = w.at[..., w_idx].set(hardplus(new_val[..., interval_idx]))
        for c_idx, new_val in intervention_params.c.items():
            c = c.at[..., c_idx].set(hardplus(new_val[..., interval_idx]))

        return y, w, c

class PiecewiseAddConstantIntervention(PiecewiseIntervention):
    @jit
    def apply(self, key, y, w, c, interval_idx, intervention_params):
        for y_idx, new_val in intervention_params.y.items():
            y = y.at[..., y_idx].add(new_val[..., interval_idx])
        for w_idx, new_val in intervention_params.w.items():
            w = w.at[..., w_idx].add(new_val[..., interval_idx])
        for c_idx, new_val in intervention_params.c.items():
            c = c.at[..., c_idx].add(new_val[..., interval_idx])

        y = jnp.maximum(y, 0.0)
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
        self.y0 = y0
        self.w0 = w0
        self.c = c
        self.t0 = t0
        self.deltaT = deltaT

        vmap_axes = (0 if y0.ndim == 2 else None, 0 if w0.ndim == 2 else None, 0 if c.ndim == 2 else None, None, None)
        self.grn_step = vmap(grn_step, in_axes=vmap_axes, out_axes=vmap_axes[:-1])


        out_shape = DictTree()
        out_shape.y = self.y0.shape + (n_steps, )
        out_shape.w = self.w0.shape + (n_steps, )
        out_shape.c = self.c.shape + (n_steps,)
        out_shape.times = (n_steps, )

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

        @jit
        def f(carry, x):
            (key, y, w, c, t) = carry

            # make grn step
            y, w, c, t = self.grn_step(y, w, c, t, self.deltaT)

            # apply intervention
            key, subkey = jrandom.split(key)
            y, w, c = intervention_fn(key, y, w, c, t, intervention_params)

            # apply perturbation
            key, subkey = jrandom.split(key)
            y, w, c = perturbation_fn(key, y, w, c, t, perturbation_params)

            return (key, y, w, c, t), (y, w, c)

        (key, y, w, c, t), (ys, ws, cs) = lax.scan(f, (key, self.y0, self.w0, self.c, self.t0), jnp.arange(self.n_steps))
        ys = jnp.moveaxis(ys, 0, -1)
        ws = jnp.moveaxis(ws, 0, -1)
        cs = jnp.moveaxis(cs, 0, -1)
        
        outputs = DictTree()
        outputs.y = ys
        outputs.w = ws
        outputs.c = cs
        outputs.times = jnp.arange(0, 0 + self.n_steps * self.deltaT, self.deltaT)

        return outputs

class GRNRolloutStatisticsEncoder(BaseRolloutStatisticsEncoder):
    filter_fn: Callable
    update_fn: Callable
    
    def __init__(self, y_shape, time_window=jnp.r_[-100:0], is_stable_std_epsilon=1e-2, is_converging_filter_size=50,
                 is_periodic_max_frequency_threshold=40, deltaT=0.1
                 ):
        out_shape = DictTree()
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

        self.filter_fn = jit(jtu.Partial(lambda system_outputs: system_outputs.y))
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