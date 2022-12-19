from functools import partial
from jax import jit, lax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import PyTree
from typing import Callable

@partial(jit, static_argnums=(1, 2))
def filter(pytree: PyTree, filter_fn: Callable, out_treedef: jtu.PyTreeDef):
        sub_pytree = filter_fn(pytree)
        out_flat, _ = jtu.tree_flatten(sub_pytree)
        return out_treedef.unflatten(out_flat)

@partial(jit, static_argnums=(1, 2, 3))
def filter_update(pytree: PyTree, filter_fn: Callable, update_fn: Callable, out_treedef: jtu.PyTreeDef):
    sub_pytree = filter_fn(pytree)
    updated_sub_pytree = update_fn(sub_pytree)
    out_flat, _ = jtu.tree_flatten(updated_sub_pytree)
    return out_treedef.unflatten(out_flat)


def uniform(key, low: PyTree, high: PyTree, out_treedef: jtu.PyTreeDef, out_shape: PyTree, out_dtype: PyTree):
    key = out_treedef.unflatten(jrandom.split(key, out_treedef.num_leaves))
    return jtu.tree_map(
        lambda key, low, high, shape, dtype: low + (high - low) * jrandom.uniform(key, shape=shape, dtype=dtype),
        key, low, high, out_shape, out_dtype)


def normal(key, mean: PyTree, std: PyTree, out_treedef: jtu.PyTreeDef, out_shape: PyTree, out_dtype: PyTree):
    key = out_treedef.unflatten(jrandom.split(key, out_treedef.num_leaves))
    return jtu.tree_map(
        lambda key, mean, std, shape, dtype: mean + std * jrandom.normal(key, shape=shape, dtype=dtype),
        key, mean, std, out_shape, out_dtype)

@partial(jit, static_argnums=(2, ))
def nearest_neighbors(Y, X, k):
    """
    Arguments:
    Y: Array [Ny, D]
    X: Array[Nx, D
    k: int

    Returns:
    X_nearest_ids: Array[Ny, k] - matrix of the  k closest point ids in X (for each target point in Y),
    distances: Array[Ny, k] - corresponding distances
    """
    *_, Dy = Y.shape
    *_, Dx = X.shape
    assert Dy == Dx, "Points in X and Y must lie in the same D-dimensional space"

    distance_matrix = jnp.sum((Y[..., jnp.newaxis, :] - X[jnp.newaxis, ...])**2, axis=-1).squeeze()
    nearest_distances_reverse, X_nearest_ids = lax.top_k(jnp.reciprocal(distance_matrix), k)
    return X_nearest_ids, jnp.reciprocal(nearest_distances_reverse)

@jit
def hardplus(x):
    return jnp.maximum(x, 0.)

@jit
def softplus(x):
    return jnp.log(1.0 + jnp.exp(x))*100.0

@jit
def inv_softplus(x):
    return jnp.log(jnp.exp(x/100.0) - 1.0)

@jit
def flat_top_gaussian(x, x0, sigma, A, P):
    return A*jnp.exp(-((x-x0)**2/(2*sigma**2))**P)

@jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

@jit
def calc_segment_intersection(seg1_start, seg1_end, seg2_start, seg2_end):
    seg1 = seg1_end - seg1_start
    seg1_perp = jnp.array([-seg1[1], seg1[0]])
    seg2 = seg2_end - seg2_start

    v1 = seg1_start - seg2_start
    t1 = jnp.cross(seg2, v1) / jnp.dot(seg2, seg1_perp)
    t2 = jnp.dot(v1, seg1_perp) / jnp.dot(seg2, seg1_perp)

    p = seg1_start + t1 * seg1

    return p, t1, t2

@jit
def wall_sticky_collision(traj_start, traj_end, wall_start, wall_end, **kwargs):
    """
    Returns the new trajectory endpoint when the trajectory is colliding with a "sticky" wall.

    Parameters
    ---------
    traj_start, traj_end, wall_start, wall_end: points (2D or 3D array)

    Returns
    ---------
    t1:
        intersection "time" (0<=t<=1) along traj segment if there is an intersection
        NaN if traj and wall segments are colinear or have no intersection
    p:
        new traj_end point after collision
        traj_end if traj and wall segments are colinear or have no intersection
    """
    p, t1, t2 = calc_segment_intersection(traj_start, traj_end, wall_start, wall_end)
    return lax.cond((t1 >= 0.0) & (t1 <= 1.0) & (t2 >= 0.0) & (t2 <= 1.0), lambda: (t1, p), lambda: (float("nan")*jnp.ones_like(t1), traj_end))


@jit
def wall_elastic_collision(traj_start, traj_end, wall_start, wall_end, **kwargs):
    """
    Returns the new trajectory endpoint if the trajectory is colliding with a wall (assuming elastic collision).

    Parameters
    ---------
    traj_start, traj_end, wall_start, wall_end: points (2D or 3D array)

    Returns
    ---------
    t1:
        intersection "time" (0<=t<=1) along traj segment if there is an intersection
        NaN if traj and wall segments are colinear or have no intersection
    p:
        new traj_end point after collision
        traj_end if traj and wall segments are colinear or have no intersection
    """
    p, t1, t2 = calc_segment_intersection(traj_start, traj_end, wall_start, wall_end)

    traj_parallel = traj_end-traj_start
    wall_parallel = wall_end - wall_start
    wall_perp = jnp.array([-wall_parallel[1], wall_parallel[0]])

    v_parallel = jnp.dot(traj_parallel, wall_parallel) * wall_parallel / jnp.linalg.norm(wall_parallel)**2
    v_perp = jnp.dot(traj_parallel, wall_perp) * wall_perp / jnp.linalg.norm(wall_perp)**2

    p = traj_start + t1*traj_parallel + (1-t1) * (-v_perp+v_parallel)

    return lax.cond((t1 >= 0.0) & (t1 <= 1.0) & (t2 >= 0.0) & (t2 <= 1.0), lambda: (t1, p), lambda: (float("nan")*jnp.ones_like(t1), traj_end))


@jit
def calc_perpendicular_wall_distance(p, wall_start, wall_end, sigma):
    """
    Returns
    ---------
    d:
        distance of p to wall
    n:
        normal of wall (oriented toward p)
    """
    wall_parallel = wall_end - wall_start
    wall_perp = jnp.array([-wall_parallel[1], wall_parallel[0]])

    v1 = p - wall_start
    d = jnp.abs(jnp.cross(wall_parallel, v1) / jnp.linalg.norm(wall_parallel))
    n_sign = jnp.sign(jnp.dot(wall_perp, v1))
    n_sign = lax.cond(n_sign == 0, lambda: jnp.array(1.0), lambda: n_sign)
    n = n_sign * wall_perp / jnp.linalg.norm(wall_perp)

    return d, n, sigma[0]

@jit
def calc_radial_wall_distance(p, wall_start, wall_end, sigma):
    """
    Returns
    ---------
    d:
        distance of p to wall
    n:
        normal of wall (oriented toward p)
    """
    d_to_wall_extremities = jnp.array([((wall_start-p)**2).sum(), ((wall_end-p)**2).sum()])
    closest_extremity_idx = jnp.argmin(d_to_wall_extremities)
    d = jnp.sqrt(d_to_wall_extremities[closest_extremity_idx])
    closest_extremity = lax.switch(closest_extremity_idx, [lambda: wall_start, lambda: wall_end])
    other_extremity = lax.switch(closest_extremity_idx, [lambda: wall_end, lambda: wall_start])
    n = lax.cond(d == 0, lambda: (p-other_extremity)/jnp.linalg.norm(p-other_extremity), lambda: (p - closest_extremity) / d)


    # wall_dir = (wall_end-wall_start)/jnp.linalg.norm(wall_end-wall_start)
    # l = jnp.abs(jnp.dot(n, wall_dir))
    # sigma = l * sigma[1] + (1-l) * sigma[0]

    return d, n, sigma[1]

@jit
def wall_force_field_collision(traj_start, traj_end, wall_start, wall_end, sigma=jnp.array([0.5, 0.1])):
    """
    Returns the new trajectory endpoint if the trajectory is colliding with a wall that emits a repulsing force field.

    Parameters
    ---------
    traj_start, traj_end, wall_start, wall_end: points (2D or 3D array)

    Returns
    ---------
    d:
        distance of traj_start to wall
    p:
        new traj_end point after collision
    """
    is_traj_start_above_wall_start = jnp.sign(jnp.dot(wall_end-wall_start, traj_start-wall_start)) >= 0
    is_traj_start_below_wall_end = jnp.sign(jnp.dot(wall_start - wall_end, traj_start - wall_end)) >= 0

    d_start, wall_normal, sigma = lax.cond(is_traj_start_above_wall_start & is_traj_start_below_wall_end,
                                    calc_perpendicular_wall_distance, calc_radial_wall_distance,
                                    traj_start, wall_start, wall_end, sigma)

    traj_parallel = traj_end - traj_start
    wall_parallel = jnp.array([-wall_normal[1], wall_normal[0]])
    wall_parallel /= jnp.linalg.norm(wall_parallel)

    v_parallel = jnp.dot(traj_parallel, wall_parallel) * wall_parallel
    v_perp = jnp.dot(traj_parallel, wall_normal) * wall_normal

    alpha = -flat_top_gaussian(d_start, x0=0, sigma=sigma, A=2, P=1) #alpha is between (-2, 0)
    is_going_toward = -jnp.sign(jnp.dot(v_perp, wall_normal))
    f_perp = is_going_toward*alpha*v_perp
    dt = 1
    v_perp = v_perp + f_perp*dt
    v = v_perp+v_parallel
    p = traj_start + v * dt

    t1, p_prime = wall_elastic_collision(traj_start, p, wall_start, wall_end)

    return lax.cond(jnp.isnan(t1),
                    lambda: (d_start, p),
                    lambda: (d_start, p_prime))