import autodiscjax as adx
from autodiscjax.modules.sgdoptimizer import SGDOptimizer
from autodiscjax.utils.misc import filter, nearest_neighbors, normal, uniform
import equinox as eqx
from jax import jit
import jax.tree_util as jtu
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PyTree
from typing import Callable

class BaseGenerator(adx.Module):
    @jit
    def __call__(self, key):
        raise NotImplementedError

class UniformRandomGenerator(BaseGenerator):
    uniform_fn: Callable

    def __init__(self, out_treedef, out_shape, out_dtype, low, high):
        super().__init__(out_treedef, out_shape, out_dtype)
        self.uniform_fn = jit(jtu.Partial(uniform, low=low, high=high, out_treedef=self.out_treedef, out_shape=self.out_shape, out_dtype=self.out_dtype))

    def __call__(self, key):
        return self.uniform_fn(key)


class NormalRandomGenerator(BaseGenerator):
    normal_fn: Callable

    def __init__(self, out_treedef, out_shape, out_dtype, mean, std):
        super().__init__(out_treedef, out_shape, out_dtype)
        self.normal_fn = jit(jtu.Partial(normal, mean=mean, std=std, out_treedef=self.out_treedef, out_shape=self.out_shape, out_dtype=self.out_dtype))

    def __call__(self, key):
        return self.normal_fn(key)


class BaseGoalEmbeddingEncoder(adx.Module):
    @jit
    def __call__(self, key, system_outputs):
        raise NotImplementedError


class FilterGoalEmbeddingEncoder(BaseGoalEmbeddingEncoder):
    filter_fn: Callable

    def __init__(self, out_treedef, out_shape, out_dtype, filter_fn):
        BaseGoalEmbeddingEncoder.__init__(self, out_treedef, out_shape, out_dtype)
        self.filter_fn = filter_fn

    @jit
    def __call__(self, key, system_outputs):
        return filter(system_outputs, self.filter_fn, self.out_treedef)


class BaseGoalGenerator(adx.Module):
    @jit
    def __call__(self, key, reached_goal_embedding_library, system_rollout_statistics_library=None):
        raise NotImplementedError

class HypercubeGoalGenerator(BaseGoalGenerator):
    hypercube_scaling: float = eqx.static_field()

    def __init__(self, out_treedef, out_shape, out_dtype, hypercube_scaling=1.5):
        super().__init__(out_treedef, out_shape, out_dtype)
        self.hypercube_scaling = hypercube_scaling

    @jit
    def __call__(self, key, reached_goal_embedding_library, system_rollout_statistics_library=None):
        library_min = jtu.tree_map(lambda x: x.min(), reached_goal_embedding_library)
        library_max = jtu.tree_map(lambda x: x.max(), reached_goal_embedding_library)
        hypercube_center = jtu.tree_map(lambda min, max: (min+max/2.0), library_min, library_max)
        hypercube_size = jtu.tree_map(lambda min, max: (max-min) * self.hypercube_scaling, library_min, library_max)
        low = jtu.tree_map(lambda center, size: center-size/2.0, hypercube_center, hypercube_size)
        high = jtu.tree_map(lambda center, size: center+size/2.0, hypercube_center, hypercube_size)

        return uniform(key, low, high, self.out_treedef, self.out_shape, self.out_dtype)


class BaseGCInterventionSelector(adx.Module):
    @jit
    def __call__(self, key, target_goals_embeddings, reached_goal_embedding_library, system_rollout_statistics_library):
        raise NotImplementedError


class NearestNeighborInterventionSelector(BaseGCInterventionSelector):
    k: int = eqx.static_field()

    def __init__(self, out_treedef, out_shape, out_dtype, k):
        super().__init__(out_treedef, out_shape, out_dtype)
        self.k = k

    @jit
    def __call__(self, key, target_goals_embeddings, reached_goal_embedding_library, system_rollout_statistics_library=None):
        target_goals_flat, target_goals_treedef = jtu.tree_flatten(target_goals_embeddings)
        target_goals_flat = jnp.concatenate(target_goals_flat, axis=-1)
        reached_goals_flat, reached_goals_treedef = jtu.tree_flatten(reached_goal_embedding_library)
        reached_goals_flat = jnp.concatenate(reached_goals_flat, axis=-1)

        selected_intervention_ids, distances = nearest_neighbors(target_goals_flat, reached_goals_flat, k=self.k)
        selected_intervention_ids = selected_intervention_ids.squeeze()

        return selected_intervention_ids



class BaseGCInterventionOptimizer(adx.Module):
    @jit
    def __call__(self, key, intervention_fn, interventions_params, system_rollout, goal_embedding_encoder, goal_achievement_loss, target_goals_embeddings):
        raise NotImplementedError

class SGDInterventionOptimizer(BaseGCInterventionOptimizer, SGDOptimizer):

    def __init__(self, out_treedef, out_shape, out_dtype, n_optim_steps, lr):
        BaseGCInterventionOptimizer.__init__(self, out_treedef, out_shape, out_dtype)
        SGDOptimizer.__init__(self, n_optim_steps, lr)

    def __call__(self, key, intervention_fn, interventions_params, system_rollout, goal_embedding_encoder, goal_achievement_loss, target_goals_embeddings):

        @jit
        def loss_fn(key, params):
            key, subkey = jrandom.split(key)
            system_outputs = system_rollout(subkey, intervention_fn, params, None, None)

            # represent outputs -> goals
            key, subkey = jrandom.split(key)
            reached_goals_embeddings = goal_embedding_encoder(subkey, system_outputs)

            return goal_achievement_loss(reached_goals_embeddings, target_goals_embeddings)

        return SGDOptimizer.__call__(self, key, interventions_params, loss_fn)


class BaseSystemRollout(adx.Module):
    @jit
    def __call__(self, key,
                 intervention_fn=None, intervention_params=None,
                 perturbation_fn=None, perturbation_params=None):
        raise NotImplementedError


class BaseRolloutStatisticsEncoder(adx.Module):
    @jit
    def __call__(self, key, system_outputs):
        raise NotImplementedError


