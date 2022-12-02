from autodiscjax.modules.sgdoptimizer import SGDOptimizer, OpenESOptimizer
from autodiscjax.modules.eaoptimizer import EAOptimizer
import equinox as eqx
import importlib
from jax import lax, nn, vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import sbmltoodejax
from tempfile import NamedTemporaryFile
import time


class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jnp.ndarray

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = lax.scan(f, hidden, input)
        # sigmoid because we're performing binary classification
        return nn.sigmoid(self.linear(out) + self.bias)


def test_sgd_optimizer():

    key = jrandom.PRNGKey(0)

    key, subkey = jrandom.split(key)
    biomodel_id = 10
    n_system_steps = 10
    biomodel_odejax_file = NamedTemporaryFile(suffix=".py")
    biomodel_xml_body = sbmltoodejax.biomodels_api.get_content_for_model(biomodel_id)
    model_data = sbmltoodejax.parse.ParseSBMLFile(biomodel_xml_body)
    sbmltoodejax.modulegeneration.GenerateModel(model_data, biomodel_odejax_file.name)
    spec = importlib.util.spec_from_file_location("ModelSpec", biomodel_odejax_file.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grnmodel_cls = getattr(module, "ModelRollout")
    y0 = getattr(module, "y0")
    grn_system_rollout = jtu.Partial(grnmodel_cls(), n_system_steps)
    grn_system = lambda y0: grn_system_rollout(jnp.maximum(y0, 0.0))[0][:, -1].max() #maximum last value
    grn_params = y0
    key, subkey = jrandom.split(key)
    grn_targets = 100*jrandom.uniform(subkey, shape=(1, ))


    rnn_system = RNN(in_size=2, out_size=1, hidden_size=16, key=subkey)
    key, subkey = jrandom.split(key)
    rnn_params = jrandom.uniform(subkey, shape=(n_system_steps, 2,))
    key, subkey = jrandom.split(key)
    rnn_targets = jrandom.uniform(subkey, shape=(1, ))

    for system, params, target in zip([grn_system, rnn_system], [grn_params, rnn_params], [grn_targets, rnn_targets]):

        L2loss = lambda k, x, g: ((system(x) - g) ** 2).sum()
        loss_fn = jtu.Partial(L2loss, g=target)

        ea_optimizer = EAOptimizer(10, n_workers=100, noise_std=jtu.tree_map(lambda p: 1.0, params))
        ea_start = time.time()
        ea_optimized_params = ea_optimizer(key, params, loss_fn)
        ea_end = time.time()

        sgd_optimizer = SGDOptimizer(10, lr=jtu.tree_map(lambda p: 1.0, params))
        sgd_start = time.time()
        sgd_optimized_params = sgd_optimizer(key, params, loss_fn)
        sgd_end = time.time()


        openes_optimizer = OpenESOptimizer(10, lr=jtu.tree_map(lambda p: 1.0, params), n_workers=100, noise_std=jtu.tree_map(lambda p: 1.0, params))
        openes_start = time.time()
        openes_optimized_params = openes_optimizer(key, params, loss_fn)
        openes_end = time.time()


        print(f"Target: {target}, starting pos: {system(params)}, n_system_steps: {n_system_steps}\n"
              f"reached pos after EA: {system(ea_optimized_params)} , compute time: {ea_end-ea_start}\n"
              f"reached pos after SGD: {system(sgd_optimized_params)} , compute time: {sgd_end-sgd_start}\n"
              f"reached pos after OpenES: {system(openes_optimized_params)} , compute time: {openes_end-openes_start}\n")

        assert loss_fn(0, ea_optimized_params) < loss_fn(0, params)
        assert loss_fn(0, sgd_optimized_params) < loss_fn(0, params)
        assert loss_fn(0, openes_optimized_params) < loss_fn(0, params)
