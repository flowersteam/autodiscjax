from autodiscjax.experiment_pipelines import run_imgep_evaluation
import evaluation_config
import experiment_config
import exputils.data.logging as log
from create_modules import *
import time

if __name__ == "__main__":
    # Load history of interventions from the experiment
    experiment_history = DictTree.load(experiment_config.get_pipeline_config().experiment_data_save_folder + "experiment_history.pickle")
    experiment_intervention_params_library = experiment_history.intervention_params_library
    experiment_system_output_library = experiment_history.system_output_library

    # Set batch size to the length of experiment data
    experiment_config.batch_size = len(jtu.tree_leaves(experiment_intervention_params_library)[0])
    evaluation_config.batch_size = experiment_config.batch_size

    # Create System Modules
    system_rollout_config = experiment_config.get_system_rollout_config()
    system_rollout_config.n_system_steps += 1000
    system_rollout = create_system_rollout_module(system_rollout_config)
    rollout_statistics_encoder_config = experiment_config.get_rollout_statistics_encoder_config()
    rollout_statistics_encoder = create_rollout_statistics_encoder_module(system_rollout, rollout_statistics_encoder_config)

    # Create Intervention Modules
    intervention_config = experiment_config.get_intervention_config()
    _, intervention_fn = create_intervention_module(intervention_config)

    # Create Perturbation Modules
    perturbation_config = evaluation_config.get_perturbation_config()
    perturbation_generator, perturbation_fn = create_perturbation_module(perturbation_config)

    # Run Evaluation Pipeline
    pipeline_config = evaluation_config.get_pipeline_config()
    log.clear()
    log.set_directory(pipeline_config.evaluation_logging_save_folder)
    tstart = time.time()
    run_imgep_evaluation(pipeline_config.jax_platform_name, pipeline_config.seed,
                         pipeline_config.n_perturbations, pipeline_config.evaluation_data_save_folder,
                         experiment_system_output_library, experiment_intervention_params_library, intervention_fn,
                         perturbation_generator, perturbation_fn,
                         system_rollout, rollout_statistics_encoder,
                         out_sanity_check=True, save_modules=False)
    tend = time.time()
    print(tend - tstart)
    log.add_value("evaluation_time", tend - tstart)
    log.save()