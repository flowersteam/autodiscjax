from addict import Dict
import jax.numpy as jnp

class ExperimentConfig:

    def __init__(self):
        self.observed_node_ids = [0, 1]
        self.ymin = jnp.array([91.93025, 100.43926, 15.655377, 53.970665])
        self.ymax = jnp.array([7866.401, 4771.8975, 1377.4222, 1086.2181])
        self.batch_size = 10

    def get_pipeline_config(self):
        config = Dict()
        config.jax_platform_name = "cpu"
        config.seed = 0
        config.n_random_batches = 1
        config.n_imgep_batches = 4
        config.batch_size = self.batch_size
        config.experiment_data_save_folder = f"experiment_data/"
        config.experiment_logging_save_folder = f"experiment_logging/"
        return config

    def get_system_rollout_config(self):
        config = Dict()
        config.biomodel_id = 29
        config.model_filepath = "biomodel_29.py"
        config.atol = 1e-6
        config.rtol = 1e-12
        config.mxstep = 1000
        config.deltaT = 0.1
        config.batch_size = self.batch_size
        config.n_system_steps = 25000
        return config

    def get_rollout_statistics_encoder_config(self):
        config = Dict()
        config.is_stable_std_epsilon = 1e-3 * (self.ymax - self.ymin)
        config.is_converging_ratio_threshold = 0.8
        config.is_periodic_max_frequency_threshold = 40
        return config

    def get_intervention_config(self):
        config = Dict()
        config.controlled_intervals = [[0, 1e-5]]
        config.controlled_node_ids = list(range(len(self.ymin)))

        config.low = Dict()
        config.high = Dict()
        for y_idx in config.controlled_node_ids:
            config.low.y[y_idx] = self.ymin[y_idx]
            config.high.y[y_idx] = self.ymax[y_idx]

        config.batch_size = self.batch_size
        return config

    def get_perturbation_config(self):
        config = Dict()
        config.perturbation_type = "null"
        return config

    def get_goal_embedding_encoder_config(self):
        config = Dict()
        config.observed_node_ids = self.observed_node_ids
        config.batch_size = self.batch_size
        return config

    def get_goal_generator_config(self):
        config = Dict()
        config.low = 0.0
        config.high = None

        # config.generator_type = "hypercube"
        # config.hypercube_scaling = 1.3

        config.generator_type = "IMFlow"
        optimizer_config = self.get_gc_intervention_optimizer_config()
        config.IM_val_scaling = 20.0
        config.IM_grad_scaling = 0.1
        config.random_proba = 0.2
        config.flow_noise = 0.1
        config.time_window = jnp.r_[-self.batch_size * optimizer_config.n_optim_steps * optimizer_config.n_workers:0]

        return config

    def get_goal_achievement_loss_config(self):
        config = Dict()
        return config

    def get_gc_intervention_selector_config(self):
        config = Dict()
        config.k = 1
        config.batch_size = self.batch_size
        return config

    def get_gc_intervention_optimizer_config(self):
        config = Dict()

        # config.optimizer_type = "EA"
        # config.n_optim_steps = 1
        # config.n_workers = 3
        # config.init_noise_std = 0.1

        config.optimizer_type = "SGD"
        config.n_optim_steps = 3
        config.n_workers = 1
        config.init_noise_std = 0.
        config.lr = 0.1

        return config