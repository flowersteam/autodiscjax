from autodiscjax import DictTree
from autodiscjax.experiment_pipelines import run_imgep_experiment
import autodiscjax.modules.imgepwrappers as imgep
import autodiscjax.modules.grnwrappers as grn
import importlib
import jax.numpy as jnp
import jax.tree_util as jtu
import sbmltoodejax

jax_platform_name = "cpu"
seed = 0
biomodel_id = 29
biomodel_odejax_filepath = f"biomodel_{biomodel_id}.py"
n_random_batches = 1
n_imgep_batches = 4
n_system_steps = 200
n_optim_steps = 5
lr = 0.2
batch_size = 100
hypercube_scaling = 1.5
rmin = 0.1
rmax = 10.0
deltaT = 0.1
atol = 1e-6
rtol = 1e-12
mxstep = 50000
save_folder = "experiment_data/"


# Prepare PyTree structures/shape/dtype
intervention_fn = grn.PiecewiseSetConstantIntervention(time_to_interval_fn=grn.TimeToInterval(intervals=[[0,1], [5,6]]))
intervention_params_tree = DictTree()
intervention_params_tree.y[0] = "placeholder"
intervention_params_tree.y[1] = "placeholder"
intervention_params_treedef = jtu.tree_structure(intervention_params_tree)
intervention_params_shape = jtu.tree_map(lambda _: (2, ), intervention_params_tree)
batched_intervention_params_shape = jtu.tree_map(lambda shape: (batch_size, )+shape, intervention_params_shape, is_leaf=lambda node: isinstance(node, tuple))
intervention_params_dtype = jtu.tree_map(lambda _: jnp.float32, intervention_params_tree)

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

intervention_selector_tree = "placeholder"
intervention_selector_treedef = jtu.tree_structure(intervention_selector_tree)
batched_intervention_selector_shape = jtu.tree_map(lambda _: (batch_size, ), intervention_selector_tree)
intervention_selector_dtype = jtu.tree_map(lambda _: jnp.int32, intervention_selector_tree)

# Create Modules
intervention_params_low = DictTree()
intervention_params_low.y[0] = rmin*198.5128
intervention_params_low.y[1] = rmin*0.
intervention_params_high = DictTree()
intervention_params_high.y[0] = rmax*800.
intervention_params_high.y[1] = rmax*387.24646
intervention_low = jtu.tree_map(lambda val, shape, dtype: val*jnp.ones(shape=shape, dtype=dtype), intervention_params_low, batched_intervention_params_shape, intervention_params_dtype)
intervention_high = jtu.tree_map(lambda val, shape, dtype: val*jnp.ones(shape=shape, dtype=dtype), intervention_params_high, batched_intervention_params_shape, intervention_params_dtype)
random_intervention_generator = imgep.UniformRandomGenerator(intervention_params_treedef, batched_intervention_params_shape, intervention_params_dtype,
                                                                         intervention_low, intervention_high)

perturbation_mean = jtu.tree_map(lambda shape, dtype: jnp.zeros(shape=shape, dtype=dtype), perturbation_params_shape, perturbation_params_dtype, is_leaf=lambda node: isinstance(node, tuple))
perturbation_std = jtu.tree_map(lambda shape, dtype: jnp.zeros(shape=shape, dtype=dtype), perturbation_params_shape, perturbation_params_dtype, is_leaf=lambda node: isinstance(node, tuple))
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
goal_generator = imgep.HypercubeGoalGenerator(goal_embedding_treedef, batched_goal_embedding_shape, goal_embedding_dtype, hypercube_scaling)

k=1
gc_intervention_selector = imgep.NearestNeighborInterventionSelector(intervention_selector_treedef, batched_intervention_selector_shape, intervention_selector_dtype, k)

def goal_achievement_loss(reached_goals_embeddings, target_goals_embeddings):
    return jnp.square(reached_goals_embeddings-target_goals_embeddings).sum()

gc_intervention_optimizer = imgep.SGDInterventionOptimizer(intervention_params_treedef, batched_intervention_params_shape, intervention_params_dtype,
                                                           n_optim_steps, jtu.tree_map(lambda low, high: lr*(high-low), intervention_params_low, intervention_params_high))


goal_filter_fn = jtu.Partial(lambda system_outputs: system_outputs.y[:, (2, 3), -1])
goal_embedding_encoder = imgep.FilterGoalEmbeddingEncoder(goal_embedding_treedef, batched_goal_embedding_shape, goal_embedding_dtype, goal_filter_fn)


# Run Experiment
run_imgep_experiment(jax_platform_name, seed, n_random_batches, n_imgep_batches, save_folder,
                     random_intervention_generator, intervention_fn,
                     perturbation_generator, perturbation_fn,
                     system_rollout, rollout_statistics_encoder,
                     goal_generator, gc_intervention_selector, gc_intervention_optimizer,
                     goal_embedding_encoder, goal_achievement_loss, out_sanity_check=True)