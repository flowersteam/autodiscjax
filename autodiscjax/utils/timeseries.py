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
def is_converging(x, time_window=jnp.r_[-1000:0], filter_size=50):
    """
    x is a signal of shape ...xT
    """
    diff = jnp.diff(x[..., time_window])
    def smooth(x):
        smooth_filter = jnp.ones(filter_size) / filter_size
        return jnp.convolve(x, smooth_filter, mode='same')
    offset = filter_size//2+1
    for _ in range(0, diff.ndim-1):
        smooth = vmap(smooth)
    diff_smooth = smooth(diff)[..., offset:-offset]
    is_diff_monotonous, diff_slope_sign = is_monotonous(jnp.abs(diff_smooth), time_window=jnp.r_[-diff_smooth.shape[-1]:0])
    is_converging = is_diff_monotonous & (diff_slope_sign <= 0)

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