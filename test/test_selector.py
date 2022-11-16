from autodiscjax.modules.sgdoptimizer import SGDOptimizer
import jax.numpy as jnp
import jax.random as jrandom

def test_nearest_neighbors():
    key = jrandom.PRNGKey(0)

    sgdoptimizer = SGDOptimizer()