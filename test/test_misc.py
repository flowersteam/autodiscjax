from autodiscjax.utils.misc import wall_sticky_collision, wall_elastic_collision, wall_force_field_collision
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def test_wall_elastic_collision():
    # no intersection
    assert jnp.isnan(
        wall_elastic_collision(jnp.array([0., 0.]), jnp.array([1., 1.]), jnp.array([2., 2.]), jnp.array([3., 3.]))[0]).all()
    assert jnp.isnan(
        wall_elastic_collision(jnp.array([0., 0.]), jnp.array([1., 1.]), jnp.array([1.1, 1.1]), jnp.array([0.1, 0.1]))[0]).all()

    # colinear
    assert jnp.isnan(
        wall_elastic_collision(jnp.array([0., 0.]), jnp.array([1., 1.]), jnp.array([0., 0.]), jnp.array([1., 1.]))[0]).all()
    assert jnp.isnan(
        wall_elastic_collision(jnp.array([0., 0.]), jnp.array([1., 1.]), jnp.array([1., 1.]), jnp.array([2., 2.]))[0]).all()



    # intersection
    assert wall_elastic_collision(jnp.array([0., 0.]), jnp.array([1., 1.]), jnp.array([1., 1.]),
                                 jnp.array([0., 2.]))[0] == 1.0
    assert (wall_elastic_collision(jnp.array([0., 0.]), jnp.array([1., 1.]), jnp.array([1., 1.]),
                                 jnp.array([0., 2.]))[1] == jnp.array([1., 1.])).all()
    assert wall_elastic_collision(jnp.array([0., 0.]), jnp.array([1., 1.]), jnp.array([0., 1.]),
                                jnp.array([1., 0.]))[0] == 0.5
    assert (((wall_elastic_collision(jnp.array([0., 0.]), jnp.array([1., 1.]), jnp.array([0., 1.]),
                                 jnp.array([1., 0.]))[1] - jnp.array([0., 0.]))**2).sum() < 1e-3).all()
    assert (((wall_elastic_collision(jnp.array([0., 0.]), jnp.array([2., 1.]), jnp.array([1., 0.]),
                             jnp.array([1., 1.]))[1] - jnp.array([0., 1.])) ** 2).sum() < 1e-3).all()

    key = jrandom.PRNGKey(0)
    for _ in range(10):
        key, subkey = jrandom.split(key)
        A, B, C, D = jrandom.uniform(subkey, shape=(4,2))
        assert (((wall_elastic_collision(A, B, C, D)[1] - wall_elastic_collision(A, B, D, C)[1]) ** 2).sum() < 1e-3).all()
        assert (jnp.dot((wall_elastic_collision(A, B, C, D)[1] - wall_elastic_collision(B, A, C, D)[1]), (D-C)) -
                 jnp.dot(B-A, D-C)) < 1e-3


def test_wall_force_field_collision():
    key = jrandom.PRNGKey(0)
    A, B, C, D = jnp.array([0., 0.]), jnp.array([1., 1.]), jnp.array([0., 0.]), jnp.array([1., 1.])
    for _ in range(100):
        key, subkey = jrandom.split(key)
        sigma = 5*jrandom.uniform(subkey, shape=(2, ))
        d, p = wall_force_field_collision(A, B, C, D, sigma=sigma)
        d2, p2 = wall_force_field_collision(A, B, D, C, sigma=sigma)
        assert (d - d2) < 1e-3 and ((p - p2) ** 2).sum() < 1e-3
        assert (d-0) < 1e-3 and ((p-jnp.array([1., 1.]))**2).sum() < 1e-3

        d, p = wall_force_field_collision(jnp.array([0.5, 0.5]), jnp.array([1.5, 0.5]), jnp.array([1., 0.]), jnp.array([1., 1.]), sigma=sigma)
        assert (d-0.5) < 1e-3 and (p[1]-0.5) < 1e-3 and (-0.5 < p[0]) and (p[0] < 1.0)


def test_visualize_wall_collision(variant="force_field"):
    key = jrandom.PRNGKey(0)

    dt = 0.2
    n = 50
    L = 10
    sigma = jnp.array([L/10, L/40])
    w = 1 / 3
    n_steps = 800

    key, subkey = jrandom.split(key)
    particles_pos = jrandom.uniform(subkey, minval=0, maxval=L, shape=(n, 2))
    particles_f = jrandom.uniform(subkey, minval=-2, maxval=2., shape=(n, 2))
    particles_v = jnp.zeros((n, 2)) + particles_f * dt

    key, subkey = jrandom.split(key)
    wall_start = L * jrandom.uniform(subkey, shape=(2,))
    key, subkey = jrandom.split(key)
    wall_end = wall_start + 2 * w * L * jrandom.uniform(subkey, shape=(2,)) - w * L

    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(xlim=(0, L), ylim=(0, L))
    scatter = ax.scatter(particles_pos[:, 0], particles_pos[:, 1])
    ax.plot([wall_start[0], wall_end[0]], [wall_start[1], wall_end[1]])
    plt.axis("off")

    all_positions = [particles_pos]
    for step_idx in range(n_steps):
        if variant == "force_field":
            _, particles_pos = vmap(wall_force_field_collision, in_axes=(0, 0, None, None, None))(particles_pos,
                                                                                                  particles_pos + particles_v * dt,
                                                                                                  wall_start, wall_end,
                                                                                                  sigma)
        elif variant == "elastic":
            _ , particles_pos = vmap(wall_elastic_collision, in_axes=(0, 0, None, None))(particles_pos, particles_pos + particles_v * dt, wall_start, wall_end)

        elif variant == "sticky":
            _, particles_pos = vmap(wall_sticky_collision, in_axes=(0, 0, None, None))(particles_pos, particles_pos + particles_v * dt, wall_start, wall_end)

        particles_pos = particles_pos % L
        all_positions.append(particles_pos)

    def update(i):
        scatter.set_offsets(all_positions[i])
        return scatter,

    anim = FuncAnimation(fig, update, frames=n_steps + 1, interval=50)
    plt.show()


