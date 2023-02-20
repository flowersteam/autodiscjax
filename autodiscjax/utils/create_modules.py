from autodiscjax import DictTree
import autodiscjax.modules.imgepwrappers as imgep
import autodiscjax.modules.grnwrappers as grn
from autodiscjax.modules import optimizers
from autodiscjax.utils.misc import wall_elastic_collision, wall_force_field_collision
import importlib
from jax import vmap
import jax.numpy as jnp
import jax.tree_util as jtu

def create_system_rollout_module(system_rollout_config):
    if system_rollout_config.system_type == "grn":
        spec = importlib.util.spec_from_file_location("JaxBioModelSpec", system_rollout_config.model_filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        grnstep_cls = getattr(module, "ModelStep")
        grnstep = grnstep_cls(atol=system_rollout_config.atol,
                              rtol=system_rollout_config.rtol,
                              mxstep=system_rollout_config.mxstep)
        y0 = getattr(module, "y0")
        w0 = getattr(module, "w0")
        c = getattr(module, "c")
        t0 = getattr(module, "t0")
        system_rollout = grn.GRNRollout(n_steps=system_rollout_config.n_system_steps, y0=y0, w0=w0, c=c, t0=t0,
                                        deltaT=system_rollout_config.deltaT, grn_step=grnstep)

    else:
        raise ValueError
    return system_rollout

def create_rollout_statistics_encoder_module(rollout_statistics_encoder_config):
    if rollout_statistics_encoder_config.statistics_type == "null":
        rollout_statistics_encoder = imgep.NullRolloutStatisticsEncoder()

    elif rollout_statistics_encoder_config.statistics_type == "grn":
        rollout_statistics_encoder = grn.GRNRolloutStatisticsEncoder(y_shape=rollout_statistics_encoder_config.y_shape,
                                                                     is_stable_time_window=rollout_statistics_encoder_config.is_stable_time_window,
                                                                     is_stable_std_epsilon=rollout_statistics_encoder_config.is_stable_std_epsilon,
                                                                     is_converging_time_window=rollout_statistics_encoder_config.is_converging_time_window,
                                                                     is_converging_ratio_threshold=rollout_statistics_encoder_config.is_converging_ratio_threshold,
                                                                     is_monotonous_time_window=rollout_statistics_encoder_config.is_monotonous_time_window,
                                                                     is_periodic_time_window=rollout_statistics_encoder_config.is_periodic_time_window,
                                                                     is_periodic_max_frequency_threshold=rollout_statistics_encoder_config.is_periodic_max_frequency_threshold,
                                                                     is_periodic_deltaT=rollout_statistics_encoder_config.is_periodic_deltaT)
    else:
        raise ValueError
    return rollout_statistics_encoder

def create_intervention_module(intervention_config):
    if intervention_config.intervention_type == "set_uniform":
        intervention_fn = grn.PiecewiseSetConstantIntervention(
            time_to_interval_fn=grn.TimeToInterval(intervals=intervention_config.controlled_intervals))
        intervention_params_tree = DictTree()
        for y_idx in intervention_config.controlled_node_ids:
            intervention_params_tree.y[y_idx] = "placeholder"
        intervention_params_treedef = jtu.tree_structure(intervention_params_tree)
        intervention_params_shape = jtu.tree_map(lambda _: (len(intervention_config.controlled_intervals),), intervention_params_tree)
        intervention_params_dtype = jtu.tree_map(lambda _: jnp.float32, intervention_params_tree)

        intervention_low = DictTree(intervention_config.low)
        intervention_low = jtu.tree_map(lambda val, shape, dtype: val * jnp.ones(shape=shape, dtype=dtype),
                                        intervention_low, intervention_params_shape,
                                        intervention_params_dtype)
        intervention_high = DictTree(intervention_config.high)
        intervention_high = jtu.tree_map(lambda val, shape, dtype: val * jnp.ones(shape=shape, dtype=dtype),
                                         intervention_high, intervention_params_shape,
                                         intervention_params_dtype)
        random_intervention_generator = imgep.UniformRandomGenerator(intervention_params_treedef,
                                                                     intervention_params_shape,
                                                                     intervention_params_dtype,
                                                                     intervention_low, intervention_high)
    else:
        raise ValueError
    return random_intervention_generator, intervention_fn

def create_perturbation_module(perturbation_config):
    if perturbation_config.perturbation_type == "null":
        perturbation_fn = grn.NullIntervention()
        perturbation_params_tree = "placeholder"
        perturbation_params_treedef = jtu.tree_structure(perturbation_params_tree)
        perturbation_params_dtype = jtu.tree_map(lambda _: jnp.float32, perturbation_params_tree)
        perturbation_params_shape = jtu.tree_map(lambda _: (0, ), perturbation_params_tree)

        perturbation_generator = imgep.EmptyArrayGenerator(perturbation_params_treedef, perturbation_params_shape, perturbation_params_dtype)

    elif perturbation_config.perturbation_type in ["noise", "push"]:
        perturbation_fn = grn.PiecewiseAddConstantIntervention(
            time_to_interval_fn=grn.TimeToInterval(intervals=perturbation_config.perturbed_intervals))

        perturbation_params_tree = DictTree()
        for y_idx in perturbation_config.perturbed_node_ids:
            perturbation_params_tree.y[y_idx] = "placeholder"
        perturbation_params_treedef = jtu.tree_structure(perturbation_params_tree)
        perturbation_params_shape = jtu.tree_map(lambda _: (len(perturbation_config.perturbed_intervals),),
                                                 perturbation_params_tree)
        perturbation_params_dtype = jtu.tree_map(lambda _: jnp.float32, perturbation_params_tree)

        if perturbation_config.perturbation_type == "noise":
            perturbation_generator = grn.NoisePerturbationGenerator(perturbation_params_treedef,
                                                                    perturbation_params_shape,
                                                                    perturbation_params_dtype,
                                                                    std=perturbation_config.std)

        elif perturbation_config.perturbation_type == "push":
            perturbation_generator = grn.PushPerturbationGenerator(perturbation_params_treedef,
                                                                    perturbation_params_shape,
                                                                    perturbation_params_dtype,
                                                                    amplitude=perturbation_config.amplitude)

    elif perturbation_config.perturbation_type == "wall":
        if perturbation_config.wall_type == "elastic":
            collision_fn = jtu.Partial(vmap(wall_elastic_collision, in_axes=(None, None, 0, 0), out_axes=(0, 0)))
        elif perturbation_config.wall_type == "force_field":
            collision_fn = jtu.Partial(vmap(wall_force_field_collision, in_axes=(None, None, 0, 0), out_axes=(0, 0)))
        perturbation_fn = grn.PiecewiseWallCollisionIntervention(
            time_to_interval_fn=grn.TimeToInterval(intervals=perturbation_config.perturbed_intervals), collision_fn=collision_fn)

        perturbation_params_tree = DictTree()
        for y_idx in perturbation_config.perturbed_node_ids:
            perturbation_params_tree.y[y_idx] = "placeholder"
        perturbation_params_tree.sigma = "placeholder"

        perturbation_params_treedef = jtu.tree_structure(perturbation_params_tree)
        n_walls = perturbation_config.n_walls
        perturbation_params_shape = jtu.tree_map(lambda _: (n_walls, 2, len(perturbation_config.perturbed_intervals),),
                                                 perturbation_params_tree)
        perturbation_params_dtype = jtu.tree_map(lambda _: jnp.float32, perturbation_params_tree)


        perturbation_generator = grn.WallPerturbationGenerator(perturbation_params_treedef,
                                                               perturbation_params_shape,
                                                               perturbation_params_dtype,
                                                               n_walls=perturbation_config.n_walls,
                                                               intersection_windows=perturbation_config.walls_intersection_window,
                                                               length_ranges=perturbation_config.walls_length_range,
                                                               sigmas=perturbation_config.walls_sigma)

    elif perturbation_config.perturbation_type == "wall":
        raise NotImplementedError

    else:
        raise ValueError

    return perturbation_generator, perturbation_fn

def create_goal_embedding_encoder_module(goal_embedding_encoder_config):
    if goal_embedding_encoder_config.encoder_type == "filter":
        goal_embedding_encoder = imgep.FilterGoalEmbeddingEncoder(goal_embedding_encoder_config.out_treedef, goal_embedding_encoder_config.out_shape,
                                                                  goal_embedding_encoder_config.out_dtype, goal_embedding_encoder_config.filter_fn)

    else:
        raise ValueError

    return goal_embedding_encoder


def create_goal_generator_module(goal_generator_config):
    if goal_generator_config.generator_type == "uniform":
        goal_generator = imgep.UniformGoalGenerator(goal_generator_config.out_treedef,
                                                      goal_generator_config.out_shape, goal_generator_config.out_dtype,
                                                      goal_generator_config.low, goal_generator_config.high)

    elif goal_generator_config.generator_type == "hypercube":
        goal_generator = imgep.HypercubeGoalGenerator(goal_generator_config.out_treedef, goal_generator_config.out_shape, goal_generator_config.out_dtype,
                                                      goal_generator_config.low, goal_generator_config.high,
                                                      goal_generator_config.hypercube_scaling)

    elif goal_generator_config.generator_type == "IMFlow":
        goal_generator = imgep.IMFlowGoalGenerator(goal_generator_config.out_treedef, goal_generator_config.out_shape, goal_generator_config.out_dtype,
                                                   goal_generator_config.low, goal_generator_config.high,
                                                   imgep.LearningProgressIM(), goal_generator_config.IM_val_scaling, goal_generator_config.IM_grad_scaling,
                                                   goal_generator_config.random_proba, goal_generator_config.flow_noise,
                                                   goal_generator_config.time_window)
    else:
        raise ValueError

    return goal_generator


def create_goal_achievement_loss_module(goal_achievement_loss_config):
    if goal_achievement_loss_config.loss_type == "L2":
        gc_loss_tree = "placeholder"
        gc_loss_treedef = jtu.tree_structure(gc_loss_tree)
        gc_loss_shape = jtu.tree_map(lambda _: (), gc_loss_tree)
        gc_loss_dtype = jtu.tree_map(lambda _: jnp.float32, gc_loss_tree)
        goal_achievement_loss = imgep.L2GoalAchievementLoss(gc_loss_treedef, gc_loss_shape, gc_loss_dtype)

    else:
        raise ValueError
    return goal_achievement_loss

def create_gc_intervention_selector_module(gc_intervention_selector_config):
    intervention_selector_tree = "placeholder"
    intervention_selector_treedef = jtu.tree_structure(intervention_selector_tree)
    intervention_selector_shape = jtu.tree_map(lambda _: (), intervention_selector_tree)
    intervention_selector_dtype = jtu.tree_map(lambda _: jnp.int32, intervention_selector_tree)

    if gc_intervention_selector_config.selector_type == "nearest_neighbor":
        gc_intervention_selector = imgep.NearestNeighborInterventionSelector(intervention_selector_treedef,
                                                                             intervention_selector_shape,
                                                                             intervention_selector_dtype, gc_intervention_selector_config.k)
    elif gc_intervention_selector_config.selector_type == "random":
        gc_intervention_selector = imgep.RandomInterventionSelector(intervention_selector_treedef,
                                                                    intervention_selector_shape,
                                                                    intervention_selector_dtype)

    else:
        raise ValueError
    return gc_intervention_selector

def create_gc_intervention_optimizer_module(gc_intervention_optimizer_config):
    if gc_intervention_optimizer_config.optimizer_type == "SGD":
        optimizer = optimizers.SGDOptimizer(gc_intervention_optimizer_config.out_treedef,
                                            gc_intervention_optimizer_config.out_shape,
                                            gc_intervention_optimizer_config.out_dtype,
                                            gc_intervention_optimizer_config.low,
                                            gc_intervention_optimizer_config.high,
                                            gc_intervention_optimizer_config.n_optim_steps,
                                            gc_intervention_optimizer_config.n_workers,
                                            init_noise_std=gc_intervention_optimizer_config.init_noise_std,
                                            lr=gc_intervention_optimizer_config.lr,
                                            )


    elif gc_intervention_optimizer_config.optimizer_type == "EA":
        optimizer = optimizers.EAOptimizer(gc_intervention_optimizer_config.out_treedef,
                                           gc_intervention_optimizer_config.out_shape,
                                           gc_intervention_optimizer_config.out_dtype,
                                           gc_intervention_optimizer_config.low,
                                           gc_intervention_optimizer_config.high,
                                           gc_intervention_optimizer_config.n_optim_steps,
                                           gc_intervention_optimizer_config.n_workers,
                                           init_noise_std=gc_intervention_optimizer_config.init_noise_std
                                           )

    else:
        raise ValueError

    gc_intervention_optimizer = imgep.BaseGCInterventionOptimizer(optimizer)

    return gc_intervention_optimizer