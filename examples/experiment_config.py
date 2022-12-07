from addict import Dict

batch_size = 50

def get_system_rollout_config():
    config = Dict()
    config.biomodel_id = 29
    config.biomodel_odejax_filepath = "biomodel_29.py"
    config.atol = 1e-6
    config.rtol = 1e-12
    config.mxstep = 50000
    config.deltaT = 0.1
    config.batch_size = batch_size
    config.n_system_steps = 500
    return config

def get_intervention_config():
    config = Dict()
    config.controlled_intervals = [[0, 1e-5]]
    config.controlled_node_ids = [0, 1, 2, 3]

    config.low = Dict()
    config.high = Dict()
    for y_idx in config.controlled_node_ids:
        config.low.y[y_idx] = [91.93025, 100.43926, 15.655377, 53.970665][y_idx]
        config.high.y[y_idx] = [7866.401, 4771.8975, 1377.4222, 1086.2181][y_idx]

    config.batch_size = batch_size
    return config

def get_perturbation_config():
    config = Dict()
    config.perturbation_type = "null"
    return config

def get_goal_embedding_encoder_config():
    config = Dict()
    config.observed_node_ids = [2, 3]
    config.batch_size = batch_size
    return config

def get_goal_generator_config():
    config = Dict()

    # config.generator_type = "hypercube_sampling"
    # config.hypercube_scaling = 1.5

    config.generator_type = "IMFlow_sampling"
    config.IM_grad_scaling = 0.1
    config.random_popsize = 0.2
    config.selected_popsize = 0.2
    config.flow_noise = 0.1
    return config

def get_goal_achievement_loss_config():
    config = Dict()
    return config

def get_gc_intervention_selector_config():
    config = Dict()
    config.k = 1
    config.batch_size = batch_size
    return config

def get_gc_intervention_optimizer_config():
    config = Dict()

    # config.optimizer_type = "EA"
    # config.n_optim_steps = 20
    # config.lr = 0.2
    # config.n_workers = 50
    # config.noise_std = 0.1

    config.optimizer_type = "SGD"
    config.n_optim_steps = 20
    config.lr = 0.2

    # config.optimizer_type = "OpenES"
    # config.n_optim_steps = 20
    # config.lr = 0.2
    # config.n_workers = 100
    # config.noise_std = 0.2

    return config

def get_pipeline_config():
    config = Dict()
    config.jax_platform_name = "cpu"
    config.seed = 0
    config.n_random_batches = 1
    config.n_imgep_batches = 4
    config.experiment_data_save_folder = "experiment_data/"
    config.experiment_logging_save_folder = "experiment_logging/"
    return config
