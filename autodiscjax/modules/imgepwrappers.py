import autodiscjax as adx
from autodiscjax.modules.optimizers import BaseOptimizer
from autodiscjax.utils.misc import filter, nearest_neighbors, normal, uniform
import equinox as eqx
import exputils.data.logging as log
from functools import partial
from jax import jit, lax, value_and_grad, vmap
import jax.tree_util as jtu
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PyTree
from typing import Callable

class BaseGenerator(adx.Module):
    low: PyTree = None
    high: PyTree = None

    @jit
    def __call__(self, key):
        raise NotImplementedError

    @jit
    def clamp(self, out_pytree, is_leaf=None):
        return jtu.tree_map(lambda val, low, high: jnp.minimum(jnp.maximum(val, low), high), out_pytree, self.low,
                            self.high, is_leaf=is_leaf)

class EmptyArrayGenerator(adx.Module):
    @jit
    def __call__(self, key):
        return jtu.tree_map(lambda shape, dtype: jnp.empty(shape=shape, dtype=dtype), self.out_shape, self.out_dtype,
                            is_leaf=lambda node: isinstance(node, tuple))

class UniformRandomGenerator(BaseGenerator):
    uniform_fn: Callable

    def __init__(self, out_treedef, out_shape, out_dtype, low, high):
        super().__init__(out_treedef, out_shape, out_dtype, low, high)
        self.uniform_fn = jit(jtu.Partial(uniform, low=low, high=high, out_treedef=self.out_treedef, out_shape=self.out_shape, out_dtype=self.out_dtype))

    def __call__(self, key):
        return self.uniform_fn(key)

class NormalRandomGenerator(BaseGenerator):
    normal_fn: Callable

    def __init__(self, out_treedef, out_shape, out_dtype, mean, std, low=None, high=None):
        super().__init__(out_treedef, out_shape, out_dtype, low, high)
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
    hypercube_scaling: float = 1.5

    def __init__(self, out_treedef, out_shape, out_dtype, hypercube_scaling=1.5):
        super().__init__(out_treedef, out_shape, out_dtype)
        self.hypercube_scaling = hypercube_scaling

    @jit
    def __call__(self, key, target_goal_embedding_library, reached_goal_embedding_library, system_rollout_statistics_library=None):
        library_min = jtu.tree_map(lambda x: x.min(0), reached_goal_embedding_library)
        library_max = jtu.tree_map(lambda x: x.max(0), reached_goal_embedding_library)
        hypercube_center = jtu.tree_map(lambda min, max: (min+max/2.0), library_min, library_max)
        hypercube_size = jtu.tree_map(lambda min, max: (max-min) * self.hypercube_scaling, library_min, library_max)
        low = jtu.tree_map(lambda center, size: center-size/2.0, hypercube_center, hypercube_size)
        high = jtu.tree_map(lambda center, size: center+size/2.0, hypercube_center, hypercube_size)

        return uniform(key, low, high, self.out_treedef, self.out_shape, self.out_dtype)

class BaseIM(eqx.Module):
    @jit
    def __call__(self, key, target_goal_embedding_library, reached_goal_embedding_library, batch_size):
        raise NotImplementedError

class LearningProgressIM(BaseIM):
    @partial(jit, static_argnames=("batch_size", ))
    def __call__(self, key, target_goal_embedding_library, reached_goal_embedding_library, batch_size):
        target_goals = jtu.tree_map(lambda node: node[-batch_size:], target_goal_embedding_library)
        reached_goals = jtu.tree_map(lambda node: node[-batch_size:], reached_goal_embedding_library)
        previously_reached_goals = jtu.tree_map(lambda node: node[:-batch_size], reached_goal_embedding_library)

        target_goals_flat, target_goals_treedef = jtu.tree_flatten(target_goals)
        target_goals_flat = jnp.concatenate(target_goals_flat, axis=-1)
        reached_goals_flat, reached_goals_treedef = jtu.tree_flatten(reached_goals)
        reached_goals_flat = jnp.concatenate(reached_goals_flat, axis=-1)
        previously_reached_goals_flat, previously_reached_goals_treedef = jtu.tree_flatten(previously_reached_goals)
        previously_reached_goals_flat = jnp.concatenate(previously_reached_goals_flat, axis=-1)

        if len(previously_reached_goals_flat) > 0:
            closest_intervention_ids, distances = nearest_neighbors(target_goals_flat, previously_reached_goals_flat,
                                                                    k=min(len(previously_reached_goals_flat), 1))
            previously_closest_goals_flat = previously_reached_goals_flat[closest_intervention_ids.squeeze()]

            def LP(goal_points, starting_points, reached_points):
                return jnp.sqrt(jnp.square(goal_points - starting_points).sum(-1)) - jnp.sqrt(jnp.square(goal_points - reached_points).sum(-1))

            IM_vals, IM_grads = vmap(value_and_grad(LP, 0), in_axes=(0, 0, 0))(target_goals_flat, previously_closest_goals_flat, reached_goals_flat)

        else:
            IM_vals = jnp.zeros(shape=(batch_size, ), dtype=jnp.float32)
            IM_grads = jrandom.uniform(key, shape=reached_goals_flat.shape, dtype=reached_goals_flat.dtype)

        return IM_vals, IM_grads

class IMFlowGoalGenerator(BaseGoalGenerator):
    """
    IM_fn: function that returns tensors IM_val, IM_grad. Example: LearningProgressIM()
    """
    IM_fn: BaseIM
    IM_grad_scaling: float = 0.4
    random_popsize: float = 0.2
    selected_popsize: float = 0.2
    flow_noise: float = 0.1

    def __init__(self, out_treedef, out_shape, out_dtype, IM_fn=LearningProgressIM(), IM_grad_scaling=0.4, random_popsize=0.2, selected_popsize=0.2, flow_noise=0.1):
        super().__init__(out_treedef, out_shape, out_dtype)
        self.IM_fn = IM_fn
        self.IM_grad_scaling = IM_grad_scaling
        self.random_popsize = random_popsize
        self.selected_popsize = selected_popsize
        self.flow_noise = flow_noise

    #@eqx.filter_jit
    def __call__(self, key, target_goal_embedding_library, reached_goal_embedding_library, system_rollout_statistics_library=None):
        out_batched_shape_flat, _ = jtu.tree_flatten(self.out_shape, is_leaf=lambda node: isinstance(node, tuple))
        batch_size = out_batched_shape_flat[0][0]

        # Generate random goals
        random_popsize = int(batch_size * self.random_popsize)
        random_out_shape = jtu.tree_map(lambda shape: (random_popsize, ) + shape[1:], self.out_shape, is_leaf=lambda node: isinstance(node, tuple))
        library_min = jtu.tree_map(lambda x: x.min(0), reached_goal_embedding_library)
        library_max = jtu.tree_map(lambda x: x.max(0), reached_goal_embedding_library)
        key, subkey = jrandom.split(key)
        random_next_goals = uniform(subkey, library_min, library_max, self.out_treedef, random_out_shape, self.out_dtype)

        # Select goals with highest IM, duplicate them, and flow them along IM_grad (with some flow noise)
        selected_popsize = int(batch_size * self.selected_popsize)

        key, subkey = jrandom.split(key)
        IM_vals, IM_grads = self.IM_fn(subkey, target_goal_embedding_library, reached_goal_embedding_library, batch_size)
        log.add_value('IM_vals', IM_vals)
        log.add_value('IM_grads', IM_grads)

        selected_pop_ids = jnp.argsort(IM_vals)[-selected_popsize:]
        n_repeats = (batch_size - random_popsize) // selected_popsize
        n_add = (batch_size - random_popsize) - n_repeats * selected_popsize
        flowed_pop_ids = jnp.concatenate([jnp.repeat(selected_pop_ids, n_repeats), selected_pop_ids[-n_add:0]])
        flowed_out_shape = jtu.tree_map(lambda shape: (len(flowed_pop_ids),) + shape[1:], self.out_shape,
                                        is_leaf=lambda node: isinstance(node, tuple))
        grad_updates = self.IM_grad_scaling * IM_grads[flowed_pop_ids]
        grad_updates_flat, _ = jtu.tree_flatten(grad_updates)
        grad_updates = self.out_treedef.unflatten(grad_updates_flat)
        #grad_updates = jtu.tree_map(lambda gu, min, max: gu*(max-min), grad_updates, library_min, library_max)
        key, subkey = jrandom.split(key)
        noise_std = jtu.tree_map(lambda min, max: self.flow_noise*(max-min), library_min, library_max)
        zero_mean = jtu.tree_map(lambda sig: jnp.zeros_like(sig), noise_std)
        noise_updates = normal(subkey, zero_mean, noise_std, self.out_treedef, flowed_out_shape, self.out_dtype)
        original_points = jtu.tree_map(lambda node: node[-batch_size:][flowed_pop_ids], reached_goal_embedding_library)
        flowed_goals = jtu.tree_map(lambda o, gu, nu: o+gu+nu, original_points, grad_updates, noise_updates)

        # concatenate
        next_goals = jtu.tree_map(lambda random, flowed: jnp.concatenate([random, flowed], axis=0), random_next_goals, flowed_goals)
        return next_goals


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
        selected_intervention_ids = jrandom.choice(key, selected_intervention_ids, axis=1)

        return selected_intervention_ids

class BaseGoalAchievementLoss(eqx.Module):
    @jit
    def __call__(self, reached_goals_embeddings, target_goals_embeddings):
        raise NotImplementedError

class L2GoalAchievementLoss(BaseGoalAchievementLoss):
    @jit
    def __call__(self, reached_goals_embeddings, target_goals_embeddings):
        return jnp.square(reached_goals_embeddings - target_goals_embeddings).sum()

class BaseGCInterventionOptimizer(adx.Module):
    optimizer: BaseOptimizer

    def __call__(self, key, intervention_fn, interventions_params, system_rollout, goal_embedding_encoder, goal_achievement_loss, target_goals_embeddings):

        @jit
        def loss_fn(key, params):
            key, subkey = jrandom.split(key)
            system_outputs = system_rollout(subkey, intervention_fn, params, None, None)

            # represent outputs -> goals
            key, subkey = jrandom.split(key)
            reached_goals_embeddings = goal_embedding_encoder(subkey, system_outputs)

            return goal_achievement_loss(reached_goals_embeddings, target_goals_embeddings)

        return self.optimizer(key, interventions_params, loss_fn)

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


