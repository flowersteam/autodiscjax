from addict import Dict
from autodiscjax import DictTree
import jax.numpy as jnp
import jax.tree_util as jtu

class ExperimentConfig:

    def __init__(self):
        self.biomodel_id = 29
        self.n_nodes = 4
        self.observed_node_ids = [2,3]
        self.ymin = jnp.array([91.93025, 100.43926, 15.655377, 53.970665])
        self.ymax = jnp.array([7866.401, 4771.8975, 1377.4222, 1086.2181])
        self.batch_size = 10
        self.T_secs = 2500
        self.model_filepath = "biomodel_29.py"

    def get_pipeline_config(self):
        config = Dict()
        config.jax_platform_name = "cpu"
        config.seed = 0
        config.n_random_batches = 1
        config.n_imgep_batches = 4
        config.batch_size = self.batch_size
        config.experiment_data_save_folder = f"experiment_data/"
        return config

    def get_system_rollout_config(self):
        config = Dict()
        config.system_type = "grn"
        config.model_filepath = self.model_filepath
        config.atol = 1e-6
        config.rtol = 1e-12
        config.mxstep = 1000
        config.deltaT = 0.1
        config.n_system_steps = int(self.T_secs / config.deltaT)
        return config

    def get_rollout_statistics_encoder_config(self):
        system_rollout_config = self.get_system_rollout_config()

        config = Dict()
        config.statistics_type = "grn"
        config.y_shape = (self.n_nodes, system_rollout_config.n_system_steps)
        config.is_stable_time_window = jnp.r_[-system_rollout_config.n_system_steps // 100:0]
        config.is_stable_std_epsilon = jnp.maximum(1e-6, 1e-2 * (self.ymax-self.ymin))
        config.is_converging_time_window = jnp.r_[-system_rollout_config.n_system_steps // 2:0]
        config.is_converging_ratio_threshold = 0.8
        config.is_monotonous_time_window = jnp.r_[-system_rollout_config.n_system_steps // 100:0]
        config.is_periodic_time_window = jnp.r_[-system_rollout_config.n_system_steps // 2:0]
        config.is_periodic_max_frequency_threshold = 40
        config.is_periodic_deltaT = system_rollout_config.deltaT
        return config

    def get_random_intervention_generator_config(self):
        config = Dict()
        config.intervention_type = "set_uniform"
        config.controlled_intervals = [[0, 1e-5]]

        config.controlled_node_ids = list(range(self.n_nodes))
        intervention_params_tree = DictTree()
        for y_idx in config.controlled_node_ids:
            intervention_params_tree.y[y_idx] = "placeholder"

        config.out_treedef = jtu.tree_structure(intervention_params_tree)
        config.out_shape = jtu.tree_map(lambda _: (len(config.controlled_intervals),),
                                                 intervention_params_tree)
        config.out_dtype = jtu.tree_map(lambda _: jnp.float32, intervention_params_tree)

        config.low = DictTree()
        config.high = DictTree()
        for y_idx in config.controlled_node_ids:
            config.low.y[y_idx] = self.ymin[y_idx] * jnp.ones(shape=config.out_shape.y[y_idx], dtype=config.out_dtype.y[y_idx])
            config.high.y[y_idx] = self.ymax[y_idx] * jnp.ones(shape=config.out_shape.y[y_idx], dtype=config.out_dtype.y[y_idx])

        return config

    def get_perturbation_config(self):
        config = Dict()
        config.perturbation_type = "null"

        return config

    def get_goal_embedding_encoder_config(self):
        config = Dict()
        config.encoder_type = "filter"
        goal_embedding_tree = "placeholder"
        config.out_treedef = jtu.tree_structure(goal_embedding_tree)
        config.out_shape = jtu.tree_map(lambda _: (len(self.observed_node_ids),), goal_embedding_tree)
        config.out_dtype = jtu.tree_map(lambda _: jnp.float32, goal_embedding_tree)
        config.filter_fn = jtu.Partial(lambda system_outputs: system_outputs.ys[..., self.observed_node_ids, -1])

        return config

    def get_goal_generator_config(self):
        config = Dict()
        config.low = 0.0
        config.high = None

        goal_embedding_encoder_config = self.get_goal_embedding_encoder_config()
        config.out_treedef = goal_embedding_encoder_config.out_treedef
        config.out_shape = goal_embedding_encoder_config.out_shape
        config.out_dtype = goal_embedding_encoder_config.out_dtype

        config.low = 0.0
        config.high = None

        # config.generator_type = "hypercube"
        # config.hypercube_scaling = 1.3

        config.generator_type = "IMFlow"
        optimizer_config = self.get_gc_intervention_optimizer_config()
        config.distance_fn = jtu.Partial(lambda y, x: jnp.sqrt(jnp.square(y - x).sum(-1)))
        config.IM_val_scaling = 20.0
        config.IM_grad_scaling = 0.1
        config.random_proba = 0.2
        config.flow_noise = 0.1
        config.time_window = jnp.r_[-self.batch_size * optimizer_config.n_optim_steps * optimizer_config.n_workers:0]

        return config

    def get_goal_achievement_loss_config(self):
        config = Dict()
        config.loss_type = "L2"
        return config

    def get_gc_intervention_selector_config(self):
        config = Dict()
        config.selector_type = "nearest_neighbor"
        config.loss_f = jtu.Partial(lambda y, x: jnp.sqrt(jnp.square(y - x).sum(-1)))
        config.k = 1
        return config

    def get_gc_intervention_optimizer_config(self):
        config = Dict()

        random_intervention_generator_config = self.get_random_intervention_generator_config()
        config.out_treedef = random_intervention_generator_config.out_treedef
        config.out_shape = random_intervention_generator_config.out_shape
        config.out_dtype = random_intervention_generator_config.out_dtype

        config.low = random_intervention_generator_config.low
        config.high = random_intervention_generator_config.high

        # config.optimizer_type = "EA"
        # config.n_optim_steps = 1
        # config.n_workers = 3
        # config.init_noise_std = jtu.tree_map(lambda low, high: 0.1*(high-low), config.low, config.high)

        config.optimizer_type = "SGD"
        config.n_optim_steps = 3
        config.n_workers = 1
        config.init_noise_std = jtu.tree_map(lambda low, high: 0.1*(high-low), config.low, config.high)
        config.lr = jtu.tree_map(lambda low, high: 0.1*(high-low), config.low, config.high)

        return config