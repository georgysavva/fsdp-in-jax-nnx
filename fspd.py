import copy
import functools
import logging

import jax.experimental
import jax.experimental.multihost_utils
from src.utils import rngs as rngs_util
from src.utils.debug import log_mem, log_shard_map
from src.utils.sharding import (
    build_shardings,
    infer_sharding,
    make_fsarray_from_local_slice,
)

logging.basicConfig(
    level=logging.INFO,  # ← INFO and above will be printed
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
import os
import typing as tp  # For type hints

import humanize
import jax

jax.distributed.initialize()  # Initialize JAX distributed
import jax.numpy as jnp  # JAX's accelerated NumPy
import numpy as np  # Standard NumPy for data generation
import optax
import orbax.checkpoint as ocp

# Import the Flax NNX API components
from flax import nnx
from flax.nnx import spmd  # new helper utilities
from flax.training import orbax_utils  # helper that builds {leaf: ArrayRestoreArgs}
from jax.debug import visualize_array_sharding  # new in 0.4.13

# Utilities for creating device meshes easily
from jax.experimental import mesh_utils, multihost_utils

# Core JAX sharding components: Mesh defines the device grid,
# PartitionSpec defines how tensor axes map to mesh axes,
# NamedSharding links PartitionSpec to a Mesh with named axes.
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# from ema_fsdp import EMA  # ← new

mesh, strategy, data_sharding, repl_sharding = build_shardings(
    mesh=[["devices", -1]],
    data_axis="devices",
    strategy=[[".*", 'fsdp(axis="devices")']],
)


class MLP(nnx.Module):
    def __init__(self, din, dmid, dout, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(din, dmid, rngs=rngs)
        self.fc2 = nnx.Linear(dmid, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(
            rate=0.1, deterministic=True
        )  # stochastic dropout layer
        self.fc3 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        x = self.fc1(x)  # Apply first layer
        x = nnx.relu(x)
        x = self.fc2(x)
        x = nnx.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)
        return x


in_features = 1
out_features = 1


def init():
    """Initialize the MLP model."""
    model = MLP(in_features, 16_384, out_features, rngs=nnx.Rngs(0))
    opt = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=1e-3, weight_decay=1e-4),
    )
    graph, state = nnx.split(opt)
    ema = copy.deepcopy(model)
    ema_state = jax.tree.map(lambda x: jnp.zeros_like(x), nnx.state(model, nnx.Param))
    nnx.update(ema, ema_state)
    ema_graph, ema_state = nnx.split(ema)
    return state, ema_state, graph, ema_graph


rngs = jax.random.PRNGKey(1)
ema_decay = 0.99

state_shape, ema_state_shape, _, _ = jax.eval_shape(init)
state_sharding = infer_sharding(state_shape, strategy, mesh)
ema_state_sharding = infer_sharding(ema_state_shape, strategy, mesh)
if jax.process_index() == 0:
    print("state sharding:", state_sharding)
    print("ema state sharding:", ema_state_sharding)
log_mem("Before initialization")
state, ema_state, graph, ema_graph = jax.jit(
    init,
    out_shardings=(state_sharding, ema_state_sharding, repl_sharding, repl_sharding),
)()
if jax.process_index() == 0:
    print("state default", state)
    print("graph default", graph)
opt = nnx.merge(graph, state)
opt.model.eval()
graph_eval, state_eval = nnx.split(opt)
if jax.process_index() == 0:
    print("stat eval", state_eval)
    print("graph eval", graph_eval)
opt.model.train()  # Set the model to training mode
graph_train, state_train = nnx.split(opt)
if jax.process_index() == 0:
    print("state train", state_train)
    print("graph train", graph_train)

options = ocp.CheckpointManagerOptions(
    save_interval_steps=1,  # this handles the control flow of how many steps to save
    max_to_keep=2,  # this handles the control flow of how many checkpoints to keep
    step_prefix="debug_toy_fsdp",
    keep_period=20000,  # this keeps step % keep_period == 0; can be used as backup
    enable_async_checkpointing=False,
    # create=False,
    # multiprocessing_options=ocp.checkpoint_manager.MultiprocessingOptions(
    #     primary_host=0,
    #     active_processes=list(range(jax.process_count())),
    # )
)
ckpt_mngr = ocp.CheckpointManager(
    "gs://us-central2-storage/solaris/dev-pd/data/ckpt/minecraft/",
    # ocp.PyTreeCheckpointer(),
    options=options,
)
latest_step = None
latest_step = ckpt_mngr.latest_step()
if latest_step is not None:
    state_restored = ckpt_mngr.restore(
        latest_step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(state),
            ema_state=ocp.args.StandardRestore(ema_state),
            rngs=ocp.args.ArrayRestore(rngs),
        ),
    )
    log_mem("After restoring state")
    state, ema_state, rngs = (
        state_restored.state,
        state_restored.ema_state,
        state_restored.rngs,
    )
    if jax.process_index() == 0:
        log_shard_map("state sharding after restore", state)
        log_shard_map("ema state sharding after restore", ema_state)
start_step = latest_step + 1 if latest_step is not None else 0

log_mem("After initialization")
if jax.process_index() == 0:
    log_shard_map("state sharding", state)
    log_shard_map("ema state sharding", ema_state)


@functools.partial(
    jax.jit, donate_argnums=(0,), out_shardings=(state_sharding, repl_sharding)
)
def train_step(
    state,
    graph,
    x,
    y,
):
    opt = nnx.merge(graph, state)

    def loss_fn(model):

        return ((model(x) - y) ** 2).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(opt.model)
    opt.update(grads)  # in-place AdamW update

    # nnx.display(rngs)
    # jax.debug.print("rngs {}", rngs)
    # r = rngs.time()
    # jax.debug.print("random normal {}", jax.random.normal(r))
    _, state = nnx.split(opt)
    return state, loss


@functools.partial(jax.jit, out_shardings=(repl_sharding, data_sharding, repl_sharding))
def test_loss_step(state: nnx.State, graph: nnx.GraphDef, step, x, y, *, rngs):
    """Compute loss without updating model (test-time)."""
    jax.debug.print("rngs at step {}: {} {}", step, rngs.shape, rngs)
    rngs, n_rngs = jax.random.split(rngs)
    n = jax.random.normal(n_rngs, shape=x.shape)
    n = jax.lax.with_sharding_constraint(n, data_sharding)
    jax.debug.visualize_array_sharding(n)
    jax.debug.print("noise at {} {} {}", step, n.shape, n)
    optimizer = nnx.merge(graph, state)
    model = optimizer.model
    model.eval()
    pred = model(x)
    loss = (pred - y) ** 2
    mean = loss.mean()
    return mean, loss, rngs


def globalize_batch(batch):
    return jax.tree.map(
        lambda x: make_fsarray_from_local_slice(x, mesh.devices.flatten()),
        batch,
    )


@functools.partial(jax.jit, donate_argnums=(1,), out_shardings=ema_state_sharding)
def update_ema(state, ema_state, graph, ema_graph):
    """Update the EMA model."""
    opt = nnx.merge(graph, state)
    model = opt.model
    ema = nnx.merge(ema_graph, ema_state)
    state, ema_state = nnx.state(model, nnx.Param), nnx.state(ema, nnx.Param)
    ema_state = jax.tree.map(
        lambda p_model, p_ema: p_ema * ema_decay + p_model * (1 - ema_decay),
        state,
        ema_state,
    )
    nnx.update(ema, ema_state)
    _, ema_state = nnx.split(ema)
    return ema_state


# A generator function to yield batches of data for training.
def dataset(batch_size, num_steps):
    for _ in range(num_steps):
        # Randomly sample indices for the batch.
        # Yield the corresponding input and target pairs.
        yield np.random.randn(batch_size, in_features), np.random.randn(
            batch_size, out_features
        )


# --- Training Loop ---
losses = []  # To store loss values for plotting
# Iterate through the dataset generator for 10,000 steps.
for step, (x_batch_local, y_batch_local) in enumerate(
    dataset(batch_size=4, num_steps=2), start=start_step
):

    log_mem("before globalize")
    x_batch = globalize_batch(x_batch_local)  # Globalize the input batch

    y_batch = globalize_batch(y_batch_local)  # Globalize the target batch

    if jax.process_index() == 0:
        print("x_batch", x_batch.sharding)
    # Execute the JIT-compiled training step with the sharded model, optimizer, and data.
    with mesh:
        log_mem("before train step")
        state, loss = train_step(state, graph, x_batch, y_batch)
        log_mem("after train step")
        mean, tloss, rngs = test_loss_step(
            state, graph, step, x_batch, y_batch, rngs=rngs
        )  # test loss without update
        log_mem("after test step")
        if jax.process_index() == 0:
            print(f"mean at step {step}", mean)
        tloss = jax.experimental.multihost_utils.process_allgather(tloss)
        ema_state = update_ema(state, ema_state, graph, ema_graph)
        if jax.process_index() == 0:
            log_shard_map("ema state sharding after update", ema_state)
        ckpt_mngr.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),  # save_args for state
                ema_state=ocp.args.StandardSave(ema_state),  # save_args for ema_state
                rngs=ocp.args.ArraySave(rngs),
            ),
        )
        log_mem("After step %d after save" % step)
