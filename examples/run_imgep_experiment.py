from autodiscjax.experiment_pipelines import run_imgep_experiment
from autodiscjax.utils.create_modules import *
import experiment_config
import exputils.data.logging as log
import importlib
import sbmltoodejax

if __name__ == "__main__":
    config = experiment_config.ExperimentConfig()

    # Create System Modules
    system_rollout_config = config.get_system_rollout_config()
    ## create sbmltoodejax files
    biomodel_xml_body = sbmltoodejax.biomodels_api.get_content_for_model(system_rollout_config.biomodel_id)
    model_data = sbmltoodejax.parse.ParseSBMLFile(biomodel_xml_body)
    sbmltoodejax.modulegeneration.GenerateModel(model_data, system_rollout_config.model_filepath)
    spec = importlib.util.spec_from_file_location("JaxBioModelSpec", system_rollout_config.model_filepath)
    ## create autodiscjax modules
    system_rollout = create_system_rollout_module(system_rollout_config)
    rollout_statistics_encoder_config = config.get_rollout_statistics_encoder_config()
    rollout_statistics_encoder = create_rollout_statistics_encoder_module(system_rollout, rollout_statistics_encoder_config)

    # Create Intervention Modules
    intervention_config = config.get_intervention_config()
    random_intervention_generator, intervention_fn = create_intervention_module(intervention_config)

    # Create Perturbation Modules
    perturbation_config = config.get_perturbation_config()
    perturbation_generator, perturbation_fn = create_perturbation_module(perturbation_config)

    # Create IMGEP modules
    ## Goal Embedding Encoder, Generator and Achievement Loss
    goal_embedding_encoder_config = config.get_goal_embedding_encoder_config()
    goal_embedding_encoder = create_goal_embedding_encoder_module(goal_embedding_encoder_config)
    goal_generator_config = config.get_goal_generator_config()
    goal_generator = create_goal_generator_module(goal_embedding_encoder, goal_generator_config)
    goal_achievement_loss_config = config.get_goal_achievement_loss_config()
    goal_achievement_loss = create_goal_achievement_loss_module(goal_achievement_loss_config)

    ## Goal-Conditioned Intervention Selector and Optimizer
    gc_intervention_selector_config = config.get_gc_intervention_selector_config()
    gc_intervention_selector = create_gc_intervention_selector_module(gc_intervention_selector_config)
    gc_intervention_optimizer_config = config.get_gc_intervention_optimizer_config()
    gc_intervention_optimizer = create_gc_intervention_optimizer_module(random_intervention_generator, gc_intervention_optimizer_config)

    # Run IMGEP Pipeline
    pipeline_config = config.get_pipeline_config()

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