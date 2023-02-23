from addict import Dict
from experiment_config import ExperimentConfig

class EvaluationConfig:

    def __init__(self):

        expe_config = ExperimentConfig()
        self.observed_node_ids = expe_config.observed_node_ids
        self.batch_size = expe_config.batch_size
        self.deltaT = expe_config.get_system_rollout_config().deltaT
        self.n_secs = expe_config.get_system_rollout_config().n_secs

    def get_perturbation_config(self):
        perturbation_type = "wall"
        return eval(f"self.get_{perturbation_type}_perturbation_config()")


    def get_noise_perturbation_config(self):
        config = Dict()
        config.perturbation_type = "noise"
        config.perturbed_intervals = [[t, t+self.deltaT/2] for t in range(100, 300, 5)]
        config.perturbed_node_ids = [0, 1, 2, 3]
        config.std = 0.01

        return config

    def get_push_perturbation_config(self):
        config = Dict()
        config.perturbation_type = "push"
        config.perturbed_intervals = [[self.n_secs/2, self.n_secs/2+self.deltaT/2]]
        config.perturbed_node_ids = [0, 1, 2, 3]
        config.magnitude = 0.1
        return config

    def get_wall_perturbation_config(self):
        config = Dict()
        config.perturbation_type = "wall"
        config.wall_type = "force_field"
        config.perturbed_intervals = [[0, 100000]]
        config.perturbed_node_ids = self.observed_node_ids
        config.n_walls = 2
        config.walls_intersection_window = [[0.1, 0.2], [0.7, 0.8]] # in distance travelled from 0 to 1.0, of size n_walls
        config.walls_length_range = [[0.2, 0.2]] * config.n_walls
        config.walls_sigma = [1e-2, 1e-4]
        return config

    def get_pipeline_config(self):
        config = Dict()
        config.jax_platform_name = "cpu"
        config.seed = 0
        config.n_perturbations = 5
        config.evaluation_data_save_folder = "evaluation_data/"
        return config
