import argparse
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import torch
from flax import nnx
from jax import random
from jax.experimental import mesh_utils
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Dataset

IN_FEATURES = 1
OUT_FEATURES = 1
HIDDEN_DIM = 1024


class SinDataset(Dataset):
    def __init__(self, seed):
        self.seed = seed
        self.reset_seed()

    def reset_seed(self):
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def __len__(self):
        return 2**31 - 1  # Large number for infinite sampling

    def __getitem__(self, idx):
        x = (
            torch.rand(1, generator=self.rng) * 2 * torch.pi - torch.pi
        )  # Random x in [-π, π]
        y = torch.sin(x)
        return x.numpy(), y.numpy()


class MLP(nnx.Module):
    def __init__(self, din, dmid, dout, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(din, dmid, rngs=rngs)
        self.fc2 = nnx.Linear(dmid, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.fc3 = nnx.Linear(dmid, dout, rngs=rngs)
        self.rngs = rngs

    def __call__(self, x, rngs):
        x = self.fc1(x)  # Apply first layer
        x = nnx.relu(x)
        x = self.fc2(x)
        x = nnx.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)
        return x, rngs


def init_ema(model: nnx.Module) -> nnx.State:
    """Initialize the Exponential Moving Average (EMA) model."""
    ema_state = jax.tree.map(lambda x: jnp.zeros_like(x), nnx.state(model))
    return ema_state


def init(learning_rate):
    """Initialize the MLP model."""
    model = MLP(
        IN_FEATURES,
        HIDDEN_DIM,
        OUT_FEATURES,
        rngs=nnx.Rngs(0, dropout=random.key(1), noise=random.key(2)),
    )
    opt = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=learning_rate),
    )
    opt_graph, opt_state = nnx.split(opt)
    ema_state = init_ema(model)
    return opt_graph, opt_state, ema_state


def create_device_mesh(axis_name):
    device_mesh = mesh_utils.create_device_mesh(
        (jax.device_count(),), devices=jax.devices()
    )
    return jax.sharding.Mesh(device_mesh, (axis_name,))


def build_shardings(data_axis: str):
    device_mesh = create_device_mesh(
        data_axis,
    )
    data_sharding = jax.sharding.NamedSharding(
        device_mesh, jax.sharding.PartitionSpec(data_axis)
    )
    repl_sharding = jax.sharding.NamedSharding(
        device_mesh, jax.sharding.PartitionSpec()
    )

    return device_mesh, data_sharding, repl_sharding


def fsdp(axis: str, cur_spec, mesh, var_state, min_size_to_shard):
    """Fully Sharded Data Parallel tactic - shard largest available dimension along given mesh axis."""
    arr = var_state.value
    if arr is None:
        return cur_spec
    shape = tuple(arr.shape)
    axis_size = mesh.shape[axis]
    if arr.size < min_size_to_shard:
        return cur_spec
    dim_indices = sorted(range(len(shape)), key=lambda i: shape[i], reverse=True)
    for i in dim_indices:
        if cur_spec[i] is None and shape[i] % axis_size == 0:
            new_spec = list(cur_spec)
            new_spec[i] = axis
            return tuple(new_spec)
    return cur_spec


def flatten_state(state: nnx.State, path: tuple[str, ...] = ()):
    """Recursively traverse an NNX VariableState, yielding (path, VariableState)."""
    if isinstance(state, nnx.VariableState):
        name = "/".join(str(p) for p in path)
        yield name, state
    elif hasattr(state, "items"):
        for key, subtree in state.items():
            yield from flatten_state(subtree, path + (key,))
    elif isinstance(state, (list, tuple)):
        for idx, subtree in enumerate(state):
            yield from flatten_state(subtree, path + (str(idx),))


def infer_sharding(
    state: nnx.State,
    mesh: jax.sharding.Mesh,
    axis: str,
    min_size_to_shard: int = 2**20,
):
    """
    Infer a sharding specification for an NNX model state by applying FSDP to every parameter.
    axis: mesh axis name to shard along. Defaults to the first mesh axis.
    """
    flat_params = list(flatten_state(state))
    vars_states = [vs for _, vs in flat_params]

    specs = [
        (None,) * vs.value.ndim if vs.value is not None else () for vs in vars_states
    ]

    for i, _ in enumerate(flat_params):
        specs[i] = fsdp(axis, specs[i], mesh, vars_states[i], min_size_to_shard)

    shardings = [
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
        for spec in specs
    ]

    sharding_tree = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(
            state, is_leaf=lambda x: isinstance(x, nnx.VariableState)
        ),
        shardings,
    )
    return sharding_tree


def log_shard_map(tag: str, state: nnx.State) -> None:
    """
    Emit one log line per *device shard* (not per tensor), e.g.

        params/fc2/kernel   shard=(slice(4096,8192), slice(None))  → TPU(0,3)
    """
    print("── Shard ↦ device map: %s ──", tag)

    for name, var in flatten_state(state):
        arr = var.value if isinstance(var, nnx.VariableState) else var
        for d, idx in arr.sharding.devices_indices_map(arr.shape).items():
            print(f" {name}  {idx}  → {d}")


def train_step(
    opt_graph: nnx.GraphDef,
    opt_state: nnx.State,
    x: jax.Array,
    y: jax.Array,
    rngs: jax.Array,
):
    """Training step for DiT on ImageNet. **All updates happened in-place.**

    Args:
    - graph: graphdef of the NNX model.
    - state: state of the NNX model.
    - rng_state: rng state for the training step.
    - batch: batch of samples and labels.
    """
    optimizer = nnx.merge(opt_graph, opt_state)
    model = optimizer.model

    def loss_fn(model, rngs: jax.Array):
        y_hat, rngs = model(x, rngs)
        loss = jnp.mean((y_hat - y) ** 2)
        return loss, rngs

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, rngs), grads = grad_fn(model, rngs=rngs)

    optimizer.update(grads)

    _, opt_state = nnx.split(optimizer)

    return opt_state, loss, rngs


def test_step(
    model_graph: nnx.GraphDef,
    model_state: nnx.State,
    x: jax.Array,
    y: jax.Array,
    rngs: jax.Array,
):
    """Compute loss without updating model (test-time)."""
    model = nnx.merge(model_graph, model_state)
    y_hat, rngs = model(x, rngs)
    loss = jnp.mean((y_hat - y) ** 2)
    # We mean-reduce across the batch dimension, then pmean across devices.
    return loss, y_hat, rngs


def make_fsarray_from_local_slice(
    local_slice: jnp.ndarray,
    global_devices: list,
    axis: str,
):
    """Create a fully-sharded global device array from local host arrays.

    Args:
        local_slice: Something convertible to a numpy array (eg also TF tensors)
        that is this host's slice of the global array.
        global_devices: The list of global devices. Needed for consistent ordering.

    Returns:
        The global on-device array which consists of all local slices stacked
        together in the order consistent with the devices.
    """
    mesh = jax.sharding.Mesh(global_devices, (axis,))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(axis))
    local_ds = mesh.local_devices

    x = np.asarray(local_slice)
    xs = jax.device_put(np.split(x, len(local_ds), axis=0), local_ds)

    global_shape = (x.shape[0] * jax.process_count(), *x.shape[1:])
    return jax.make_array_from_single_device_arrays(global_shape, sharding, xs)


def update_ema(
    model_state: nnx.State,
    ema_state: nnx.State,
    ema_decay: float,
) -> nnx.State:
    """Update the EMA model."""

    def update_param(p_model, p_ema):
        # Skip PRNG keys and only update actual parameters
        # if hasattr(p_model, "dtype") and "key" in str(p_model.dtype):
        #     return p_ema  # Return EMA PRNG key unchanged
        return p_ema * ema_decay + p_model * (1 - ema_decay)

    ema_state_no_rng = jax.tree.map(
        update_param,
        nnx.filter_state(model_state, nnx.Param),
        nnx.filter_state(ema_state, nnx.Param),
    )
    ema_state = nnx.merge_state(ema_state, ema_state_no_rng)
    return ema_state


def main(args):
    if not args.gpu:
        jax.distributed.initialize()
    print("Available JAX devices", jax.devices())

    data_axis = "data"
    mesh, data_sharding, repl_sharding = build_shardings(data_axis="data")
    init_fn = functools.partial(init, args.lr)
    _, opt_state_shape, ema_state_shape = jax.eval_shape(init_fn)
    opt_state_sharding = infer_sharding(opt_state_shape, mesh, data_axis)
    ema_state_sharding = infer_sharding(ema_state_shape, mesh, data_axis)
    if jax.process_index() == 0:
        print("Opt state sharding:", opt_state_sharding)
        print("EMA state sharding:", ema_state_sharding)

    opt_graph, opt_state, ema_state = jax.jit(
        init_fn,
        out_shardings=(repl_sharding, opt_state_sharding, ema_state_sharding),
    )()
    if jax.process_index() == 0:
        log_shard_map("Opt state sharding", opt_state)
        log_shard_map("EMA state sharding", ema_state)
        print("Opt state", opt_state)
        print("EMA state", ema_state)
    opt = nnx.merge(opt_graph, opt_state)
    opt.model.train()  # Set the model to training mode
    opt_graph, opt_state = nnx.split(opt)
    opt.model.eval()
    model_graph_eval, _ = nnx.split(opt.model)
    ckpt_mngr = ocp.CheckpointManager(
        os.path.abspath(args.checkpoint_dir),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=1,  # this handles the control flow of how many steps to save
            max_to_keep=2,  # this handles the control flow of how many checkpoints to keep
            step_prefix=args.experiment_name,
            enable_async_checkpointing=False,
            create=True,
        ),
    )

    rngs = jax.random.PRNGKey(42)
    rngs_eval = jax.random.PRNGKey(123)
    latest_step = None
    latest_step = ckpt_mngr.latest_step()
    if latest_step is not None:
        state_restored = ckpt_mngr.restore(
            latest_step,
            args=ocp.args.Composite(
                opt_state=ocp.args.StandardRestore(opt_state),
                ema_state=ocp.args.StandardRestore(ema_state),
                rngs=ocp.args.ArrayRestore(rngs),
            ),
        )
        opt_state, ema_state, rngs = (
            state_restored.opt_state,
            state_restored.ema_state,
            state_restored.rngs,
        )
        if jax.process_index() == 0:
            log_shard_map("Opt state sharding after restore", opt_state)
            log_shard_map("EMA state sharding after restore", ema_state)
            print("Opt state after restore", opt_state)
            print("EMA state after restore", ema_state)
    start_step = 0 if latest_step is None else latest_step
    local_batch_size = args.batch_size // jax.process_count()
    train_dataloader = DataLoader(
        SinDataset(seed=start_step), batch_size=local_batch_size, shuffle=False
    )
    test_dataset = SinDataset(seed=-1)
    test_dataloader = DataLoader(
        test_dataset, batch_size=local_batch_size, shuffle=False
    )

    # Initialize RNGs for training

    # Compile training and test functions
    train_step_fn = jax.jit(
        train_step,
        donate_argnums=(1,),
        out_shardings=(opt_state_sharding, repl_sharding, repl_sharding),
    )

    test_step_fn = jax.jit(
        test_step,
        out_shardings=(repl_sharding, data_sharding, repl_sharding),
    )
    update_ema_fn = jax.jit(
        update_ema,
        out_shardings=ema_state_sharding,
        donate_argnums=(1,),
    )

    # Training loop
    train_iter = iter(train_dataloader)
    ema_decay = 0.9999

    for step in range(start_step, start_step + args.steps):
        # Get training batch
        x_batch, y_batch = next(train_iter)
        x_batch = make_fsarray_from_local_slice(
            x_batch, mesh.devices.flatten(), data_axis
        )
        y_batch = make_fsarray_from_local_slice(
            y_batch, mesh.devices.flatten(), data_axis
        )

        # Training step
        opt_state, train_loss, rngs = train_step_fn(
            opt_graph, opt_state, x_batch, y_batch, rngs
        )

        # Update EMA
        ema_state = update_ema_fn(opt_state["model"], ema_state, ema_decay)

        # Log training loss
        if jax.process_index() == 0 and (step + 1) % args.log_interval == 0:
            print(f"Step {step+1}, Train Loss: {train_loss:.6f}")

        # Test evaluation
        if (step + 1) % args.test_interval == 0:
            # Reset test dataset to original seed for consistent evaluation
            test_dataset.reset_seed()
            test_iter = iter(test_dataloader)
            x_test, y_test = next(test_iter)
            x_test = make_fsarray_from_local_slice(
                x_test, mesh.devices.flatten(), data_axis
            )
            y_test = make_fsarray_from_local_slice(
                y_test, mesh.devices.flatten(), data_axis
            )

            # Test loss for main model
            test_loss, y_pred_model, _ = test_step_fn(
                model_graph_eval, opt_state["model"], x_test, y_test, rngs_eval
            )

            # Test loss for EMA model
            # ema_state_with_rngs = nnx.merge_state(opt_state["model"], ema_state)
            test_loss_ema, y_pred_ema, _ = test_step_fn(
                model_graph_eval, ema_state, x_test, y_test, rngs_eval
            )

            # Convert back to numpy for plotting
            y_pred_model = jax.experimental.multihost_utils.process_allgather(
                y_pred_model, tiled=True
            )
            y_pred_ema = jax.experimental.multihost_utils.process_allgather(
                y_pred_ema, tiled=True
            )
            x_test = jax.experimental.multihost_utils.process_allgather(
                x_test, tiled=True
            )
            y_test = jax.experimental.multihost_utils.process_allgather(
                y_test, tiled=True
            )

            if jax.process_index() == 0:
                x_plot = np.array(x_test).flatten()
                y_true_plot = np.array(y_test).flatten()
                y_pred_ema_plot = np.array(y_pred_ema).flatten()
                y_pred_model_plot = np.array(y_pred_model).flatten()

                # Sort by x for better plotting
                sort_idx = np.argsort(x_plot)
                x_plot = x_plot[sort_idx]
                y_true_plot = y_true_plot[sort_idx]
                y_pred_ema_plot = y_pred_ema_plot[sort_idx]
                y_pred_model_plot = y_pred_model_plot[sort_idx]

                # Create output directory
                experiment_output_dir = os.path.join(
                    args.output_dir, args.experiment_name
                )
                os.makedirs(experiment_output_dir, exist_ok=True)
                # Create the plot
                fig = Figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                ax.scatter(x_plot, y_true_plot, alpha=0.7, label="Ground Truth", s=20)
                ax.scatter(
                    x_plot, y_pred_model_plot, alpha=0.7, label="Model Prediction", s=20
                )
                ax.scatter(
                    x_plot, y_pred_ema_plot, alpha=0.7, label="EMA Prediction", s=20
                )
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title("Sin Function: Ground Truth vs Model vs EMA Prediction")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Save the plot
                plot_path = os.path.join(experiment_output_dir, f"eval_{step+1}.png")
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")

                print(f"Plot saved to {plot_path}")

                # Calculate and print final metrics
                if jax.process_index() == 0:
                    print(
                        f"Step {step+1}, Test Loss: {test_loss:.6f}, EMA Test Loss: {test_loss_ema:.6f}"
                    )

        # Save checkpoint
        if (step + 1) % args.save_interval == 0:
            ckpt_mngr.save(
                step + 1,
                args=ocp.args.Composite(
                    opt_state=ocp.args.StandardSave(opt_state),
                    ema_state=ocp.args.StandardSave(ema_state),
                    rngs=ocp.args.ArraySave(rngs),
                ),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="fsdp")
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    main(args)
