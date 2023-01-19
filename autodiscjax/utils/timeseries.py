from functools import partial
from jax import vmap, jit
import jax.numpy as jnp


@jit
def is_stable(x, time_window=jnp.r_[-1000:0], std_epsilon=1e-4):
    """
    x is a signal of shape ...xT
    """
    mean_vals = jnp.nanmean(x[..., time_window], -1)
    std_vals = jnp.nanstd(x[..., time_window], -1)
    is_stable = (std_vals < std_epsilon)

    return is_stable, mean_vals, std_vals


@jit
def is_monotonous(x, time_window=jnp.r_[-1000:0]):
    """
    x is a signal of shape ...xT
    """
    diff = jnp.diff(x[..., time_window])
    is_monotonous = (jnp.sign(diff) == jnp.sign(diff[..., 0])[..., jnp.newaxis]).all(-1)
    diff_signs = jnp.sign(diff[..., 0])
    return is_monotonous, diff_signs


@partial(jit, static_argnames=("filter_size"))
def is_converging(x, time_window=jnp.r_[-1000:0], phase1_timepoints=(0, 1/4), phase2_timepoints=(3/4, 1), ratio_threshold=0.5):
    """
    x is a signal of shape ...xT
    """
    x = x[..., time_window]
    n_steps = x.shape[-1]

    phase1_start_idx = int(n_steps * phase1_timepoints[0])
    phase1_end_idx = int(n_steps * phase1_timepoints[1])

    phase2_start_idx = int(n_steps * phase2_timepoints[0])
    phase2_end_idx = int(n_steps * phase2_timepoints[1])

    x_phase_1 = x[..., phase1_start_idx:phase1_end_idx]
    phase1_amplitude = x_phase_1.max(-1) - x_phase_1.min(-1)
    x_phase_2 = x[..., phase2_start_idx:phase2_end_idx]
    phase2_amplitude = x_phase_2.max(-1) - x_phase_2.min(-1)

    ratio = phase2_amplitude / phase1_amplitude
    is_converging = ratio < ratio_threshold

    return is_converging


@partial(jit, static_argnames=("max_frequency_threshold"))
def is_periodic(x, time_window=jnp.r_[-1000:0], deltaT=1, max_frequency_threshold=40):
    """
    x is a signal of shape ...xT
    """
    sp = abs(jnp.fft.fft(x[..., time_window]))[..., 1:len(time_window)//2] #only consider frequencies > 0
    freqs = jnp.fft.fftfreq(n=len(time_window), d=deltaT)[1:len(time_window)//2]
    is_periodic = (sp.max(-1) > max_frequency_threshold) & (sp.argmax(-1) > 0)
    max_frequency_vals = freqs[sp.argmax(-1)]
    mean_vals = x[..., time_window].mean(-1)
    amplitude_vals = x[..., time_window].max(-1) - x[..., time_window].min(-1)

    return is_periodic, mean_vals, amplitude_vals, max_frequency_vals