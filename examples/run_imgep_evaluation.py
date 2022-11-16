from autodiscjax import DictTree
from autodiscjax.experiment_pipelines import run_imgep_evaluation
import autodiscjax.modules.imgepwrappers as imgep
import autodiscjax.modules.grnwrappers as grn
import importlib
import jax.numpy as jnp
import jax.tree_util as jtu
import sbmltoodejax

jax_platform_name = "cpu"
seed = 0
n_perturbations = 50
biomodel_id = 29
biomodel_odejax_filepath = f"biomodel_{biomodel_id}.py"
n_system_steps = 1000
deltaT = 0.1
atol = 1e-6
rtol = 1e-12
mxstep = 50000
experiment_save_folder = "experiment_data/"
evaluation_save_folder = "evaluation_data/"

# Load history from the experiment
experiment_history = DictTree.load(experiment_save_folder+"experiment_history.pickle")
batch_size = len(experiment_history.reached_goal_embedding_library)

# Prepare PyTree structures/shape/dtype
intervention_fn = grn.PiecewiseSetConstantIntervention(time_to_interval_fn=grn.TimeToInterval(intervals=[[0,1], [5,6]]))
intervention_params_library = experiment_history.intervention_params_library

perturbation_fn = grn.PiecewiseAddConstantIntervention(time_to_interval_fn=grn.TimeToInterval(intervals=[[0,10]]))
perturbation_params_tree = DictTree()
perturbation_params_tree.c[0] = "placeholder"
perturbation_params_tree.c[1] = "placeholder"
perturbation_params_treedef = jtu.tree_structure(perturbation_params_tree)
perturbation_params_shape = jtu.tree_map(lambda _: (1, ), perturbation_params_tree)
batched_perturbation_params_shape = jtu.tree_map(lambda shape: (batch_size, )+shape, perturbation_params_shape, is_leaf=lambda node: isinstance(node, tuple))
perturbation_params_dtype = jtu.tree_map(lambda _: jnp.float32, perturbation_params_tree)

goal_embedding_tree = "placeholder"
goal_embedding_treedef = jtu.tree_structure(goal_embedding_tree)
goal_embedding_shape = jtu.tree_map(lambda _: (2, ), goal_embedding_tree)
batched_goal_embedding_shape = jtu.tree_map(lambda shape: (batch_size, )+shape, goal_embedding_shape, is_leaf=lambda node: isinstance(node, tuple))
goal_embedding_dtype = jtu.tree_map(lambda _: jnp.float32, goal_embedding_tree)

# Create Modules
perturbation_mean = jtu.tree_map(lambda shape, dtype: jnp.zeros(shape=shape, dtype=dtype), perturbation_params_shape, perturbation_params_dtype, is_leaf=lambda node: isinstance(node, tuple))
perturbation_std = jtu.tree_map(lambda shape, dtype: jnp.ones(shape=shape, dtype=dtype), perturbation_params_shape, perturbation_params_dtype, is_leaf=lambda node: isinstance(node, tuple))
perturbation_generator = imgep.NormalRandomGenerator(perturbation_params_treedef, batched_perturbation_params_shape, perturbation_params_dtype,
                                                                  perturbation_mean, perturbation_std)



biomodel_xml_body = sbmltoodejax.biomodels_api.get_content_for_model(biomodel_id)
model_data = sbmltoodejax.parse.ParseSBMLFile(biomodel_xml_body)
sbmltoodejax.modulegeneration.GenerateModel(model_data, biomodel_odejax_filepath)
spec = importlib.util.spec_from_file_location("JaxBioModelSpec", biomodel_odejax_filepath)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
grnstep_cls = getattr(module, "ModelStep")
grnstep = grnstep_cls(atol=atol, rtol=rtol, mxstep=mxstep)
y0 = getattr(module, "y0")
w0 = getattr(module, "w0")
c = getattr(module, "c")
t0 = getattr(module, "t0")
y0 = jnp.tile(y0, (batch_size, 1))
w0 = jnp.tile(w0, (batch_size, 1))
c = jnp.tile(c, (batch_size, 1))
system_rollout = grn.GRNRollout(n_steps=n_system_steps, y0=y0, w0=w0, c=c, t0=t0, deltaT=deltaT,
                                grn_step=grnstep)

rollout_statistics_encoder = grn.GRNRolloutStatisticsEncoder(y_shape=system_rollout.out_shape.y, time_window=jnp.r_[-100:0],
                                                             is_stable_std_epsilon=1e-2, is_converging_filter_size=50,
                                                             is_periodic_max_frequency_threshold=40, deltaT=deltaT)

goal_filter_fn = jtu.Partial(lambda system_outputs: system_outputs.y[:, (2, 3), -1])
goal_embedding_encoder = imgep.FilterGoalEmbeddingEncoder(goal_embedding_treedef, batched_goal_embedding_shape, goal_embedding_dtype, goal_filter_fn)


# Run Experiment
run_imgep_evaluation(jax_platform_name, seed, n_perturbations, evaluation_save_folder,
                     intervention_params_library, intervention_fn,
                     perturbation_generator, perturbation_fn,
                     system_rollout, rollout_statistics_encoder,
                     goal_embedding_encoder, out_sanity_check=True)