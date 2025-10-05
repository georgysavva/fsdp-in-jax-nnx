# FSDP in JAX NNX
FSDP is an essential technique for training large models that don't fit into a single device. There are tutorials out there on how to set it up in JAX, but none of them do it for the modern, Flax NNX API, and those tutorials don't cover the whole training recipe like checkpointing and rngs that a production code would need. 

Closing this gap, this repository contains a tutorial on how to implement FSDP in JAX using NNX modules. You can view the step-by-step guide in the notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/georgysavva/fsdp-in-jax-nnx/blob/dev/fsdp_in_jax_nnx.ipynb) or check out the complete code in the [main.py](https://github.com/georgysavva/fsdp-in-jax-nnx/blob/dev/main.py) file.

## Features
The code in this repository supports the following:

- Fully working FSDP implementation in JAX on TPU that evenly shards all weights across the devices, together with DDP
- Modern, native Flax NNX module API
- Checkpointing to disk or GCP bucket via Orbax
- Reproducible `nnx.Rngs` for noise generation and dropout
- The same checkpoint can be run on TPUs with a different number of devices
- EMA version of the model
- All model operation functions are JIT compiled
- Tested on 

## Running
Here are the instructions you need to follow to try out the `main.py` code yourself.

### Local

The project uses conda. Create the environment using:

```bash
conda env create -f environment.yml
conda activate fsdp-jax
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run on your machine with:

```bash
python main.py \
    --experiment_name="fsdp_test" \
    --checkpoint_dir="{absolute path to a dir to save checkpoints to" \
```

### GCP TPU deploy

The repository contains a convenient script to run `main.py` on GCP TPUs.

First create a shell script file that looks like this:

```shell
TPU={your GCP tpu name}
EXP_NAME=fsdp_test
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOGFILE=logs/output_${TIMESTAMP}_${EXP_NAME}.log

python run_on_tpu.py \
    --resource-name   "${TPU}" \
    --gcp-zone        "{your GCP zone}" \
    --gcp-project     "{your GCP project}" \
    --git-branch      "main" \
    --run-command "python main.py \
    --experiment_name=${EXP_NAME} \
    --checkpoint_dir={path to a GCP bucket folder}" 2>&1 | tee "$LOGFILE"
```
Now you can run your shell script file. It will execute `run_on_tpu.py`, which will download this repository onto the TPU, create the conda env, install the python dependencies, and run `main.py`. It will save the checkpoint to the GCP bucket folder and outputs to the `$HOME/outputs/` directory on the TPU machine with index 0. 
