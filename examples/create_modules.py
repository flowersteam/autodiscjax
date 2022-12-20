from autodiscjax import DictTree
import autodiscjax.modules.imgepwrappers as imgep
import autodiscjax.modules.grnwrappers as grn
from autodiscjax.modules import optimizers
from autodiscjax.utils.misc import wall_elastic_collision, wall_force_field_collision
import importlib
from jax import vmap
import jax.numpy as jnp
import jax.tree_util as jtu
import sbmltoodejax

def create_system_rollout_module(system_rollout_config):
    biomodel_xml_body = sbmltoodejax.biomodels_api.get_content_for_model(system_rollout_config.biomodel_id)
    model_data = sbmltoodejax.parse.ParseSBMLFile(biomodel_xml_body)
    sbmltoodejax.modulegeneration.GenerateModel(model_data, system_rollout_config.biomodel_odejax_filepath)
    spec = importlib.util.spec_from_file_location("JaxBioModelSpec", system_rollout_config.biomodel_odejax_filepath)
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
    return system_rollout

def create_rollout_statistics_encoder_module(system_rollout):
    rollout_statistics_encoder = grn.GRNRolloutStatisticsEncoder(y_shape=system_rollout.out_shape.ys,
                                                                 time_window=jnp.r_[-100:0],
                                                                 is_stable_std_epsilon=1e-2,
                                                                 is_converging_filter_size=50,
                                                                 is_periodic_max_frequency_threshold=40, deltaT=system_rollout.deltaT)
    return rollout_statistics_encoder

def create_intervention_module(intervention_config):
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
    return random_intervention_generator, intervention_fn

def create_perturbation_module(perturbation_config):
    if perturbation_config.perturbation_type == "null":
        perturbation_fn = grn.NullIntervention()
        perturbation_params_tree = "placeholder"
        perturbation_params_treedef = jtu.tree_structure(perturbation_params_tree)
        perturbation_params_dtype = jtu.tree_map(lambda _: jnp.float32, perturbation_params_tree)
        perturbation_params_shape = jtu.tree_map(lambda _: (0, ), perturbation_params_tree)

        perturbation_generator = imgep.EmptyArrayGenerator(perturbation_params_treedef, perturbation_params_shape, perturbation_params_dtype)

    elif perturbation_config.perturbation_type == "add":
        perturbation_fn = grn.PiecewiseAddConstantIntervention(
            time_to_interval_fn=grn.TimeToInterval(intervals=perturbation_config.perturbed_intervals))

        perturbation_params_tree = DictTree()
        for y_idx in perturbation_config.perturbed_node_ids:
            perturbation_params_tree.y[y_idx] = "placeholder"
        perturbation_params_treedef = jtu.tree_structure(perturbation_params_tree)
        perturbation_params_shape = jtu.tree_map(lambda _: (len(perturbation_config.perturbed_intervals),),
                                                 perturbation_params_tree)
        perturbation_params_dtype = jtu.tree_map(lambda _: jnp.float32, perturbation_params_tree)


        perturbation_generator = grn.NoisePerturbationGenerator(perturbation_params_treedef,
                                                             perturbation_params_shape,
                                                             perturbation_params_dtype,
                                                             std=perturbation_config.std)

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
                                                               perturbation_config.walls_target_intersection_window,
                                                               perturbation_config.walls_length_range,
                                                               perturbation_config.walls_sigma)

    elif perturbation_config.perturbation_type == "wall":
        raise NotImplementedError

    else:
        raise ValueError

    return perturbation_generator, perturbation_fn

def create_goal_embedding_encoder_module(goal_embedding_encoder_config):
    goal_embedding_tree = "placeholder"
    goal_embedding_treedef = jtu.tree_structure(goal_embedding_tree)
    goal_embedding_shape = jtu.tree_map(lambda _: (len(goal_embedding_encoder_config.observed_node_ids),), goal_embedding_tree)
    goal_embedding_dtype = jtu.tree_map(lambda _: jnp.float32, goal_embedding_tree)
    goal_filter_fn = jtu.Partial(lambda system_outputs: system_outputs.ys[..., goal_embedding_encoder_config.observed_node_ids, -1])
    goal_embedding_encoder = imgep.FilterGoalEmbeddingEncoder(goal_embedding_treedef, goal_embedding_shape,
                                                              goal_embedding_dtype, goal_filter_fn)
    return goal_embedding_encoder


def create_goal_generator_module(goal_embedding_encoder, goal_generator_config):
    if goal_generator_config.generator_type == "hypercube_sampling":
        goal_generator = imgep.HypercubeGoalGenerator(goal_embedding_encoder.out_treedef, goal_embedding_encoder.out_shape, goal_embedding_encoder.out_dtype,
                                                      goal_generator_config.low, goal_generator_config.high,
                                                      goal_generator_config.hypercube_scaling)

    elif goal_generator_config.generator_type == "IMFlow_sampling":
        goal_generator = imgep.IMFlowGoalGenerator(goal_embedding_encoder.out_treedef, goal_embedding_encoder.out_shape, goal_embedding_encoder.out_dtype,
                                                   goal_generator_config.low, goal_generator_config.high,
                                                   imgep.LearningProgressIM(), goal_generator_config.IM_val_scaling, goal_generator_config.IM_grad_scaling,
                                                   goal_generator_config.random_proba, goal_generator_config.flow_noise,
                                                   goal_generator_config.time_window)
    else:
        raise ValueError

    return goal_generator


def create_goal_achievement_loss_module(goal_achievement_loss_config):
    goal_achievement_loss = imgep.L2GoalAchievementLoss()
    return goal_achievement_loss

def create_gc_intervention_selector_module(gc_intervention_selector_config):
    intervention_selector_tree = "placeholder"
    intervention_selector_treedef = jtu.tree_structure(intervention_selector_tree)
    intervention_selector_shape = jtu.tree_map(lambda _: (), intervention_selector_tree)
    intervention_selector_dtype = jtu.tree_map(lambda _: jnp.int32, intervention_selector_tree)
    gc_intervention_selector = imgep.NearestNeighborInterventionSelector(intervention_selector_treedef,
                                                                         intervention_selector_shape,
                                                                         intervention_selector_dtype, gc_intervention_selector_config.k)
    return gc_intervention_selector

def create_gc_intervention_optimizer_module(random_intervention_generator, gc_intervention_optimizer_config):
    if gc_intervention_optimizer_config.optimizer_type == "SGD":
        optimizer = optimizers.SGDOptimizer(random_intervention_generator.out_treedef,
                                            random_intervention_generator.out_shape,
                                            random_intervention_generator.out_dtype,
                                            random_intervention_generator.low,
                                            random_intervention_generator.high,
                                            gc_intervention_optimizer_config.n_optim_steps,
                                            jtu.tree_map(
                                                lambda low, high: gc_intervention_optimizer_config.lr * (high - low),
                                                random_intervention_generator.low,
                                                random_intervention_generator.high))

    elif gc_intervention_optimizer_config.optimizer_type == "OpenES":
        optimizer = optimizers.OpenESOptimizer(random_intervention_generator.out_treedef,
                                               random_intervention_generator.out_shape,
                                               random_intervention_generator.out_dtype,
                                               random_intervention_generator.low,
                                               random_intervention_generator.high,
                                               gc_intervention_optimizer_config.n_optim_steps,
                                               jtu.tree_map(
                                                   lambda low, high: gc_intervention_optimizer_config.lr * (high - low),
                                                   random_intervention_generator.low,
                                                   random_intervention_generator.high),
                                               gc_intervention_optimizer_config.n_workers,
                                               jtu.tree_map(
                                                   lambda low, high: gc_intervention_optimizer_config.noise_std * (high - low),
                                                   random_intervention_generator.low,
                                                   random_intervention_generator.high),
                                               )

    elif gc_intervention_optimizer_config.optimizer_type == "EA":
        optimizer = optimizers.EAOptimizer(random_intervention_generator.out_treedef,
                                           random_intervention_generator.out_shape,
                                           random_intervention_generator.out_dtype,
                                           random_intervention_generator.low,
                                           random_intervention_generator.high,
                                           gc_intervention_optimizer_config.n_optim_steps,
                                           gc_intervention_optimizer_config.n_workers,
                                           jtu.tree_map(
                                               lambda low, high: gc_intervention_optimizer_config.noise_std * (
                                                           high - low),
                                               random_intervention_generator.low,
                                               random_intervention_generator.high),
                                           )

    else:
        raise ValueError

    gc_intervention_optimizer = imgep.BaseGCInterventionOptimizer(optimizer)

    return gc_intervention_optimizer