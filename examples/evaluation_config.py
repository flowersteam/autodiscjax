from addict import Dict
import jax.numpy as jnp

batch_size = None

def get_perturbation_config():
    config = Dict()

    # config.perturbation_type = "add"
    # config.perturbed_intervals = [[0, 10]]
    # config.perturbed_node_ids = [0, 1, 2, 3]
    # config.std = 0.1

    config.perturbation_type = "wall"
    config.wall_type = "force_field"
    config.perturbed_intervals = [[0, 100000]]
    config.perturbed_node_ids = [2, 3]
    config.n_walls = 2
    config.walls_target_intersection_window = jnp.r_[200:1800]
    config.walls_length_range = [0.2, 0.2]
    config.walls_sigma = [1e-2, 1e-4]
    return config

def get_pipeline_config():
    config = Dict()
    config.jax_platform_name = "cpu"
    config.seed = 0
    config.n_perturbations = 5
    config.evaluation_data_save_folder = "evaluation_data/"
    config.evaluation_logging_save_folder = "evaluation_logging/"
    return config
