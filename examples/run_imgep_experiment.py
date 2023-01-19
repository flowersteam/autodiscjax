from autodiscjax.experiment_pipelines import run_imgep_experiment
import experiment_config
import exputils.data.logging as log
from create_modules import *
import time

if __name__ == "__main__":

    # Create System Modules
    system_rollout_config = experiment_config.get_system_rollout_config()
    system_rollout = create_system_rollout_module(system_rollout_config)
    rollout_statistics_encoder_config = experiment_config.get_rollout_statistics_encoder_config()
    rollout_statistics_encoder = create_rollout_statistics_encoder_module(system_rollout, rollout_statistics_encoder_config)

    # Create Intervention Modules
    intervention_config = experiment_config.get_intervention_config()
    random_intervention_generator, intervention_fn = create_intervention_module(intervention_config)

    # Create Perturbation Modules
    perturbation_config = experiment_config.get_perturbation_config()
    perturbation_generator, perturbation_fn = create_perturbation_module(perturbation_config)

    # Create IMGEP modules
    ## Goal Embedding Encoder, Generator and Achievement Loss
    goal_embedding_encoder_config = experiment_config.get_goal_embedding_encoder_config()
    goal_embedding_encoder = create_goal_embedding_encoder_module(goal_embedding_encoder_config)
    goal_generator_config = experiment_config.get_goal_generator_config()
    goal_generator = create_goal_generator_module(goal_embedding_encoder, goal_generator_config)
    goal_achievement_loss_config = experiment_config.get_goal_achievement_loss_config()
    goal_achievement_loss = create_goal_achievement_loss_module(goal_achievement_loss_config)

    ## Goal-Conditioned Intervention Selector and Optimizer
    gc_intervention_selector_config = experiment_config.get_gc_intervention_selector_config()
    gc_intervention_selector = create_gc_intervention_selector_module(gc_intervention_selector_config)
    gc_intervention_optimizer_config = experiment_config.get_gc_intervention_optimizer_config()
    gc_intervention_optimizer = create_gc_intervention_optimizer_module(random_intervention_generator, gc_intervention_optimizer_config)

    # Run IMGEP Pipeline
    pipeline_config = experiment_config.get_pipeline_config()

    ## Run
    log.clear()
    log.set_directory(pipeline_config.experiment_logging_save_folder)
    run_imgep_experiment(pipeline_config.jax_platform_name, pipeline_config.seed,
                         pipeline_config.n_random_batches, pipeline_config.n_imgep_batches, pipeline_config.batch_size,
                         pipeline_config.experiment_data_save_folder,
                         random_intervention_generator, intervention_fn,
                         perturbation_generator, perturbation_fn,
                         system_rollout, rollout_statistics_encoder,
                         goal_generator, gc_intervention_selector, gc_intervention_optimizer,
                         goal_embedding_encoder, goal_achievement_loss,
                         out_sanity_check=False, save_modules=False, logger=log)