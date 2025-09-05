import argparse
import functools
import logging
import os
from typing import Any, Generator, Tuple

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
    """A PyTorch dataset that generates sine function data points.

    This dataset generates random x values from [-π, π] and computes y = sin(x).
    The dataset uses a seeded random number generator for reproducible results.

    Args:
        seed: Random seed for reproducible data generation.
    """

    def __init__(self, seed: int) -> None:
        """Initialize the dataset with a random seed.

        Args:
            seed: Random seed for data generation.
        """
        self.seed = seed
        self.reset_seed()

    def reset_seed(self) -> None:
        """Reset the random number generator to the initial seed.

        This is useful for ensuring reproducible evaluation data.
        """
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            A very large number representing the dataset size.
        """
        return 2**31 - 1

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single data point.

        Args:
            idx: Index (unused, but required for Dataset interface).

        Returns:
            Tuple of (x, y) where x is a random value in [-π, π] and y = sin(x).
        """
        x = torch.rand(1, generator=self.rng) * 2 * torch.pi - torch.pi
        y = torch.sin(x)
        return x.numpy(), y.numpy()


class MLP(nnx.Module):
    """A Multi-Layer Perceptron (MLP) neural network using Flax NNX.

    This is a simple feedforward neural network with two hidden layers,
    ReLU activations, and dropout regularization.

    Args:
        din: Number of input features.
        dmid: Number of hidden units in each hidden layer.
        dout: Number of output features.
        rngs: Random number generators for parameter initialization and dropout.
    """

    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs) -> None:
        """Initialize the MLP with specified dimensions.

        Args:
            din: Number of input features.
            dmid: Number of hidden units in each hidden layer.
            dout: Number of output features.
            rngs: Random number generators for parameter initialization and dropout.
        """
        self.fc1 = nnx.Linear(din, dmid, rngs=rngs)
        self.fc2 = nnx.Linear(dmid, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.fc3 = nnx.Linear(dmid, dout, rngs=rngs)
        self.rngs = rngs

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, din).

        Returns:
            Output tensor of shape (batch_size, dout).
        """
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def init_ema(model: nnx.Module) -> nnx.State:
    """Initialize exponential moving average (EMA) state for a model.

    Creates a zero-initialized state tree with the same structure as the model's state.

    Args:
        model: The neural network model to create EMA state for.

    Returns:
        EMA state with the same structure as the model state, but zero-initialized.
    """
    ema_state = jax.tree.map(lambda x: jnp.zeros_like(x), nnx.state(model))
    return ema_state


def init(learning_rate: float) -> Tuple[nnx.GraphDef, nnx.State, nnx.State]:
    """Initialize the model, optimizer, and EMA state.

    Creates a new MLP model, wraps it in an AdamW optimizer, and initializes
    the exponential moving average state.

    Args:
        learning_rate: Learning rate for the AdamW optimizer.

    Returns:
        Tuple of (optimizer_graph, optimizer_state, ema_state).
    """
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


def create_device_mesh(axis_name: str) -> jax.sharding.Mesh:
    """Create a JAX device mesh for distributed computation.

    Creates a 1D mesh using all available devices and assigns the given axis name.

    Args:
        axis_name: Name to assign to the mesh axis (e.g., 'data' for data parallelism).

    Returns:
        JAX mesh object for distributed computation.
    """
    device_mesh = mesh_utils.create_device_mesh(
        (jax.device_count(),), devices=jax.devices()
    )
    return jax.sharding.Mesh(device_mesh, (axis_name,))


def build_shardings(
    data_axis: str,
) -> Tuple[
    jax.sharding.Mesh,
    jax.sharding.NamedSharding,
    jax.sharding.NamedSharding,
]:
    """Build JAX sharding configurations for distributed computation.

    Creates a device mesh and two sharding strategies:
    - Data sharding: for sharding data across devices
    - Replicated sharding: for replicating data across all devices

    Args:
        data_axis: Name of the axis for data parallelism.

    Returns:
        Tuple of (device_mesh, data_sharding, replicated_sharding).
    """
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


def fsdp(
    axis: str,
    cur_spec: Tuple[Any, ...],
    mesh: jax.sharding.Mesh,
    var_state: nnx.VariableState,
    min_size_to_shard: int,
) -> Tuple[Any, ...]:
    """Implement Fully Sharded Data Parallel (FSDP) sharding strategy.

    Determines how to shard a parameter tensor across devices. Shards the largest
    dimension that is divisible by the number of devices and meets the minimum size requirement.

    Args:
        axis: Name of the mesh axis to shard along.
        cur_spec: Current partition specification.
        mesh: JAX device mesh.
        var_state: Variable state containing the parameter tensor.
        min_size_to_shard: Minimum tensor size to consider for sharding.

    Returns:
        Updated partition specification with sharding applied if appropriate.
    """
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


def flatten_state(
    state: nnx.State, path: Tuple[str, ...] = ()
) -> Generator[Tuple[str, nnx.VariableState], None, None]:
    """Recursively flatten a nested state tree into (name, variable_state) pairs.

    Traverses the state tree and yields each variable with its hierarchical path name.

    Args:
        state: The state tree to flatten (can be nested).
        path: Current path in the hierarchy (used for recursion).

    Yields:
        Tuples of (path_name, variable_state) for each leaf variable.
    """
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
) -> nnx.State:
    """Infer optimal sharding strategy for a model state using FSDP.

    Analyzes each parameter in the state and determines the best sharding strategy
    based on tensor size and dimensions. Creates a sharding tree that matches
    the structure of the input state.

    Args:
        state: Model state to create sharding for.
        mesh: JAX device mesh for distributed computation.
        axis: Name of the mesh axis for sharding.
        min_size_to_shard: Minimum tensor size to consider for sharding.

    Returns:
        Sharding tree with the same structure as the input state.
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
    """Log the sharding mapping of arrays to devices for debugging.

    Prints a detailed breakdown of how each parameter is sharded across devices,
    showing which array indices are stored on which devices.

    Args:
        tag: Descriptive tag for the logging output.
        state: Model state to analyze for sharding information.
    """
    logging.info(f"── Shard ↦ device map: {tag} ──")

    for name, var in flatten_state(state):
        arr = var.value if isinstance(var, nnx.VariableState) else var
        for d, idx in arr.sharding.devices_indices_map(arr.shape).items():
            logging.info(f" {name}  {idx}  → {d}")


def train_step(
    opt_graph: nnx.GraphDef,
    opt_state: nnx.State,
    x: jax.Array,
    y: jax.Array,
    add_noise: bool = False,
) -> Tuple[nnx.State, jax.Array]:
    """Perform a single training step with gradient computation and parameter update.

    Computes the forward pass, loss, gradients, and updates model parameters.
    Optionally adds noise to the target values for data augmentation.

    Args:
        opt_graph: Optimizer graph definition (static structure).
        opt_state: Optimizer state (parameters and optimizer state).
        x: Input batch of shape (batch_size, input_dim).
        y: Target batch of shape (batch_size, output_dim).
        add_noise: Whether to add noise to targets for data augmentation.

    Returns:
        Tuple of (updated_optimizer_state, loss_value).
    """
    optimizer = nnx.merge(opt_graph, opt_state)
    model = optimizer.model

    def loss_fn(model: MLP) -> jax.Array:
        y_hat = model(x)
        if add_noise:
            noise_key = model.rngs["noise"]()
            noise = jax.random.normal(noise_key, y.shape)
            y_noisy = y + noise
            loss = jnp.mean((y_hat - y_noisy) ** 2)
        else:
            loss = jnp.mean((y_hat - y) ** 2)
        return loss

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    optimizer.update(grads)

    _, opt_state = nnx.split(optimizer)

    return opt_state, loss


def test_step(
    model_graph: nnx.GraphDef,
    model_state: nnx.State,
    x: jax.Array,
    y: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Perform a single evaluation step without parameter updates.

    Computes the forward pass and loss for evaluation purposes.

    Args:
        model_graph: Model graph definition (static structure).
        model_state: Model state (parameters only, no optimizer state).
        x: Input batch of shape (batch_size, input_dim).
        y: Target batch of shape (batch_size, output_dim).

    Returns:
        Tuple of (loss_value, predictions).
    """
    model = nnx.merge(model_graph, model_state)
    y_hat = model(x)
    loss = jnp.mean((y_hat - y) ** 2)
    return loss, y_hat


def make_fsarray_from_local_slice(
    local_slice: jnp.ndarray,
    global_devices: list[jax.Device],
    axis: str,
) -> jax.Array:
    """Create a globally sharded array from a local data slice.

    Takes a local data slice and creates a globally sharded JAX array
    by distributing the data across multiple devices and processes.

    Args:
        local_slice: Local portion of the data on this process.
        global_devices: List of all devices across all processes.
        axis: Name of the axis for sharding.

    Returns:
        Globally sharded JAX array with proper device placement.
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
    """Update exponential moving average (EMA) of model parameters.

    Computes the exponential moving average using the formula:
    ema_new = ema_decay * ema_old + (1 - ema_decay) * model_param

    Args:
        model_state: Current model state with updated parameters.
        ema_state: Current EMA state to be updated.
        ema_decay: Decay factor for EMA (typically close to 1.0, e.g., 0.9999).

    Returns:
        Updated EMA state.
    """

    def update_param(p_model: jax.Array, p_ema: jax.Array) -> jax.Array:
        return p_ema * ema_decay + p_model * (1 - ema_decay)

    ema_state_no_rng = jax.tree.map(
        update_param,
        nnx.filter_state(model_state, nnx.Param),
        nnx.filter_state(ema_state, nnx.Param),
    )
    ema_state = nnx.merge_state(ema_state, ema_state_no_rng)
    return ema_state


def setup_logging() -> None:
    """Setup logging configuration for INFO level console output."""
    # Configure logging format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Setup basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler()],  # Console output only
        force=True,  # Override any existing configuration
    )


def main(args: argparse.Namespace) -> None:
    """Main training loop for distributed MLP training with FSDP.

    Implements a complete training pipeline including:
    - Distributed initialization and device mesh setup
    - Model and optimizer initialization with FSDP sharding
    - Checkpoint loading and saving
    - Training loop with EMA updates
    - Periodic evaluation and visualization

    Args:
        args: Command-line arguments containing hyperparameters and configuration.
    """
    # Setup logging
    setup_logging()
    logging.info(f"Starting training with args: {args}")

    if not args.gpu:
        assert args.checkpoint_dir.startswith(
            "gs://"
        ), "Checkpoint directory must be a GCS path"
        jax.distributed.initialize()
    logging.info(f"Available JAX devices: {jax.devices()}")

    data_axis = "data"
    mesh, data_sharding, repl_sharding = build_shardings(data_axis="data")
    init_fn = functools.partial(init, args.lr)
    _, opt_state_shape, ema_state_shape = jax.eval_shape(init_fn)
    opt_state_sharding = infer_sharding(opt_state_shape, mesh, data_axis)
    ema_state_sharding = infer_sharding(ema_state_shape, mesh, data_axis)

    opt_graph, opt_state, ema_state = jax.jit(
        init_fn,
        out_shardings=(repl_sharding, opt_state_sharding, ema_state_sharding),
    )()
    if jax.process_index() == 0:
        log_shard_map("Opt state sharding", opt_state)
        log_shard_map("EMA state sharding", ema_state)
    if jax.process_index() == 0:
        logging.info("Merging optimizer graph and state")
    opt = nnx.merge(opt_graph, opt_state)
    opt.model.train()
    opt_graph, opt_state = nnx.split(opt)
    opt.model.eval()
    model_graph_eval, _ = nnx.split(opt.model)
    ckpt_mngr = ocp.CheckpointManager(
        args.checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=args.save_interval,
            max_to_keep=2,
            create=False,
            step_prefix=args.experiment_name,
            enable_async_checkpointing=False,
        ),
    )
    if jax.process_index() == 0:
        logging.info("Checkpoint manager initialized")

    latest_step = None
    latest_step = ckpt_mngr.latest_step()
    if latest_step is not None:
        state_restored = ckpt_mngr.restore(
            latest_step,
            args=ocp.args.Composite(
                opt_state=ocp.args.StandardRestore(opt_state),
                ema_state=ocp.args.StandardRestore(ema_state),
            ),
        )
        opt_state, ema_state = (
            state_restored.opt_state,
            state_restored.ema_state,
        )
        if jax.process_index() == 0:
            logging.info("Checkpoint restored successfully")
            log_shard_map("Opt state sharding after restore", opt_state)
            log_shard_map("EMA state sharding after restore", ema_state)
    start_step = 0 if latest_step is None else latest_step
    local_batch_size = args.batch_size // jax.process_count()

    if jax.process_index() == 0:
        logging.info(f"Training configuration:")
        logging.info(f"  - Starting from step: {start_step}")
        logging.info(f"  - Total processes: {jax.process_count()}")
        logging.info(f"  - Global batch size: {args.batch_size}")
        logging.info(f"  - Local batch size: {local_batch_size}")
        logging.info(f"  - Learning rate: {args.lr}")
        logging.info(f"  - Steps to run: {args.steps}")
        logging.info(f"  - Log interval: {args.log_interval}")
        logging.info(f"  - Test interval: {args.test_interval}")
        logging.info(f"  - Save interval: {args.save_interval}")
    train_dataloader = DataLoader(
        SinDataset(seed=start_step), batch_size=local_batch_size, shuffle=False
    )
    test_dataset = SinDataset(seed=-1)
    test_dataloader = DataLoader(
        test_dataset, batch_size=local_batch_size, shuffle=False
    )

    train_step_fn = jax.jit(
        train_step,
        donate_argnums=(1,),
        static_argnums=(4,),
        out_shardings=(opt_state_sharding, repl_sharding),
    )

    test_step_fn = jax.jit(
        test_step,
        out_shardings=(repl_sharding, data_sharding),
    )
    update_ema_fn = jax.jit(
        update_ema,
        out_shardings=ema_state_sharding,
        donate_argnums=(1,),
    )

    train_iter = iter(train_dataloader)
    ema_decay = 0.9999

    for step in range(start_step, start_step + args.steps):
        x_batch, y_batch = next(train_iter)
        x_batch = make_fsarray_from_local_slice(
            x_batch, mesh.devices.flatten(), data_axis
        )
        y_batch = make_fsarray_from_local_slice(
            y_batch, mesh.devices.flatten(), data_axis
        )

        with mesh:
            opt_state, train_loss = train_step_fn(
                opt_graph, opt_state, x_batch, y_batch, args.add_noise
            )

            ema_state = update_ema_fn(opt_state["model"], ema_state, ema_decay)

        if jax.process_index() == 0 and (step + 1) % args.log_interval == 0:
            logging.info(f"Step {step+1}, Train Loss: {train_loss:.6f}")

        if (step + 1) % args.test_interval == 0:
            test_dataset.reset_seed()
            test_iter = iter(test_dataloader)
            x_test, y_test = next(test_iter)
            x_test = make_fsarray_from_local_slice(
                x_test, mesh.devices.flatten(), data_axis
            )
            y_test = make_fsarray_from_local_slice(
                y_test, mesh.devices.flatten(), data_axis
            )
            with mesh:
                test_loss, y_pred_model = test_step_fn(
                    model_graph_eval, opt_state["model"], x_test, y_test
                )

                test_loss_ema, y_pred_ema = test_step_fn(
                    model_graph_eval, ema_state, x_test, y_test
                )

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

                sort_idx = np.argsort(x_plot)
                x_plot = x_plot[sort_idx]
                y_true_plot = y_true_plot[sort_idx]
                y_pred_ema_plot = y_pred_ema_plot[sort_idx]
                y_pred_model_plot = y_pred_model_plot[sort_idx]

                experiment_output_dir = os.path.join(
                    args.output_dir, args.experiment_name
                )
                os.makedirs(experiment_output_dir, exist_ok=True)
                fig = Figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                ax.scatter(x_plot, y_true_plot, alpha=0.7, label="Ground Truth", s=20)
                ax.scatter(
                    x_plot,
                    y_pred_model_plot,
                    alpha=0.7,
                    label="Model Prediction",
                    s=20,
                )
                ax.scatter(
                    x_plot,
                    y_pred_ema_plot,
                    alpha=0.7,
                    label="EMA Prediction",
                    s=20,
                )
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title("Sin Function: Ground Truth vs Model vs EMA Prediction")
                ax.legend()
                ax.grid(True, alpha=0.3)

                plot_path = os.path.join(experiment_output_dir, f"eval_{step+1}.png")
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")

                logging.info(f"Plot saved to {plot_path}")

                if jax.process_index() == 0:
                    logging.info(
                        f"Step {step+1}, Test Loss: {test_loss:.6f}, "
                        f"EMA Test Loss: {test_loss_ema:.6f}"
                    )

        if (step + 1) % args.save_interval == 0:
            if jax.process_index() == 0:
                logging.info(f"Saving checkpoint at step {step + 1}")
            ckpt_mngr.save(
                step + 1,
                args=ocp.args.Composite(
                    opt_state=ocp.args.StandardSave(opt_state),
                    ema_state=ocp.args.StandardSave(ema_state),
                ),
            )
            if jax.process_index() == 0:
                logging.info(f"Checkpoint saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="fsdp")
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--add_noise", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
