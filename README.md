# FSDP in JAX NNX

A demonstration of **Fully Sharded Data Parallel (FSDP)** training implementation using JAX and Flax NNX. This project showcases distributed training techniques for neural networks on both TPU and GPU clusters, with a focus on parameter sharding and efficient memory utilization.

## Overview

This project implements a distributed training pipeline for a simple Multi-Layer Perceptron (MLP) that learns to approximate the sine function. While the model itself is simple, the implementation demonstrates advanced distributed training concepts including:

- **Fully Sharded Data Parallel (FSDP)** parameter distribution
- **Exponential Moving Average (EMA)** for model weights
- **Distributed checkpointing** with Orbax
- **TPU/GPU cluster training** support
- **Real-time visualization** of training progress

## Key Features

### 🚀 Distributed Training

- **FSDP Implementation**: Automatically shards model parameters across devices based on tensor size and dimensions
- **Multi-device Support**: Seamlessly works on TPU v4/v5 pods and multi-GPU setups
- **Process Coordination**: Handles multi-process synchronization and data distribution

### 📊 Advanced Training Features

- **EMA Updates**: Maintains exponential moving averages of model parameters for better generalization
- **Noise Augmentation**: Optional noise injection during training for improved robustness
- **Distributed Checkpointing**: Efficient checkpoint saving/loading across distributed devices
- **Real-time Evaluation**: Periodic model evaluation with visualization

### 🔧 Production-Ready Infrastructure

- **Google Cloud TPU Integration**: Automated TPU cluster setup and job execution
- **Flexible Configuration**: Command-line arguments for all hyperparameters
- **Comprehensive Logging**: Detailed logging of training metrics and device utilization
- **Checkpoint Management**: Automatic checkpoint rotation and recovery

### Supported Features

- Fully working FSDP implementation in JAX that evenly shards all weights across the devices together with DDP
- Uses the native Flax NNX module API
- Checkpointing to disk or GCP bucket via Orbax
- Reproducible nnx.Rngs for noise generation and dropout
- Same checkpoint can be run on TPUs with different number of devices
- EMA version of the model
- All model operation functions JIT compiled

## Installation

### Environment Setup

The project uses conda for environment management. Create the environment using:

```bash
conda env create -f environment.yml
conda activate fsdp-jax
```

### Dependencies

For **TPU training**:

```bash
pip install -r requirements_tpu.txt
```

For **GPU training**:

```bash
pip install -r requirements_gpu.txt
```

For **CPU/local development**:

```bash
pip install -r requirements.txt
```

### Core Dependencies

- **JAX**: 0.5.1 (with TPU or CUDA support)
- **Flax**: 0.10.4 (NNX API)
- **Optax**: 0.2.4 (optimizers)
- **Orbax**: 0.11.16 (checkpointing)
- **PyTorch**: 2.7.0 (data loading)

## Usage

### Local Training

For quick testing on local hardware:

```bash
python main.py \
    --experiment_name="local_test" \
    --gpu \
    --steps=1000 \
    --batch_size=64 \
    --lr=1e-4 \
    --checkpoint_dir="./checkpoints" \
    --output_dir="./outputs"
```

### TPU Training

For distributed TPU training, use the provided TPU runner script:

```bash
# Edit the configuration in experiments/1_test_tpu.sh
# Then run:
bash experiments/1_test_tpu.sh
```

Or use the TPU runner directly:

```bash
python run_on_tpu.py \
    --resource-name="your-tpu-name" \
    --gcp-zone="us-central2-b" \
    --gcp-project="your-project" \
    --git-branch="main" \
    --run-command="python main.py --experiment_name=fsdp_experiment --steps=10000 --checkpoint_dir=gs://your-bucket/checkpoints"
```

### Key Parameters

| Parameter           | Description                                        | Default  |
| ------------------- | -------------------------------------------------- | -------- |
| `--experiment_name` | Name for the experiment (used in checkpoints/logs) | `"fsdp"` |
| `--steps`           | Number of training steps                           | `20000`  |
| `--batch_size`      | Global batch size (distributed across devices)     | `256`    |
| `--lr`              | Learning rate for AdamW optimizer                  | `1e-5`   |
| `--test_interval`   | Steps between evaluations                          | `1000`   |
| `--save_interval`   | Steps between checkpoint saves                     | `1000`   |
| `--add_noise`       | Enable noise augmentation during training          | `False`  |
| `--gpu`             | Use GPU instead of TPU                             | `False`  |

## Architecture Details

### Model Architecture

The MLP consists of:

- **Input Layer**: 1D input (x coordinate)
- **Hidden Layers**: Two 1024-unit layers with ReLU activation
- **Dropout**: 10% dropout rate for regularization
- **Output Layer**: 1D output (sin(x) prediction)

### FSDP Implementation

The FSDP sharding strategy:

1. **Analyzes tensors**: Identifies parameters above the minimum sharding threshold
2. **Selects dimensions**: Chooses the largest dimension divisible by device count
3. **Creates shardings**: Distributes parameters across the device mesh
4. **Maintains structure**: Preserves model structure while enabling distributed computation

### Training Pipeline

1. **Data Generation**: On-the-fly sine function data generation
2. **Forward Pass**: Distributed model evaluation
3. **Loss Computation**: Mean squared error with optional noise augmentation
4. **Backward Pass**: Distributed gradient computation
5. **Parameter Update**: FSDP-aware parameter updates
6. **EMA Update**: Exponential moving average maintenance

## Monitoring and Visualization

### Training Outputs

- **Loss Curves**: Real-time training and validation loss tracking
- **Prediction Plots**: Periodic visualization of model predictions vs. ground truth
- **Device Utilization**: Logging of parameter distribution across devices

### Checkpointing

- **Automatic Saving**: Checkpoints saved at configurable intervals
- **State Recovery**: Full training state restoration including optimizer and EMA states
- **Distributed Storage**: Supports both local and Google Cloud Storage backends

## File Structure

```
├── main.py                 # Main training script with FSDP implementation
├── run_on_tpu.py          # TPU cluster job runner
├── main_fake.py           # Testing/debugging version
├── fsdp_in_jax_nnx.ipynb  # Jupyter notebook for exploration
├── experiments/           # Experiment configurations
│   └── 1_test_tpu.sh     # TPU training script
├── requirements*.txt      # Environment dependencies
├── environment.yml        # Conda environment specification
├── local/                 # Local development scripts
├── logs/                  # Training logs
├── outputs/               # Generated plots and visualizations
└── checkpoints/           # Model checkpoints
```

## Advanced Usage

### Custom Sharding Strategies

Modify the `fsdp()` function in `main.py` to implement custom parameter sharding logic:

```python
def custom_fsdp_strategy(axis, cur_spec, mesh, var_state, min_size):
    # Your custom sharding logic here
    return updated_spec
```

### Experiment Configuration

Create new experiment scripts in the `experiments/` directory following the pattern in `1_test_tpu.sh`.

### Monitoring Training

Monitor training progress through:

- Console logs with loss metrics
- Generated plots in `outputs/` directory
- Checkpoint files for training state

## Contributing

This project serves as a reference implementation for FSDP in JAX. Contributions are welcome, particularly:

- Additional sharding strategies
- Support for other model architectures
- Performance optimizations
- Documentation improvements

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{fsdp-jax-nnx,
  title={FSDP in JAX NNX: Distributed Training with Fully Sharded Data Parallel},
  author={Georgy Savva},
  year={2024},
  url={https://github.com/georgysavva/fsdp-in-jax-nnx}
}
```

## License

This project is open source and available under the MIT License.

## Related Work

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Guide](https://flax.readthedocs.io/en/latest/nnx/index.html)
- [Orbax Checkpointing](https://orbax.readthedocs.io/)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
