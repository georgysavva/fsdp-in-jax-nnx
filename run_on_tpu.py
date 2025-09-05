import argparse
import os
import shlex
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Run on TPU")

    # Resource configuration
    parser.add_argument(
        "--resource-name", type=str, required=True, help="Name of the TPU resource"
    )
    parser.add_argument(
        "--gcp-zone",
        type=str,
        required=True,
        help="GCP zone for the TPU resource",
    )
    parser.add_argument(
        "--gcp-project",
        type=str,
        required=True,
        help="GCP project for the TPU resource",
    )
    parser.add_argument(
        "--no-setup", action="store_true", default=False, help="No setup"
    )
    # Git configuration
    parser.add_argument(
        "--git-repo-url",
        type=str,
        default="https://github.com/georgysavva/fsdp-in-jax-nnx.git",
        help="Git repository URL",
    )
    parser.add_argument(
        "--git-branch", type=str, default="master", help="Git branch to pull"
    )
    parser.add_argument(
        "--git-commit", type=str, default=None, help="Git commit to checkout"
    )
    parser.add_argument(
        "--run-command", type=str, help="Command to run on the TPU", required=True
    )

    args = parser.parse_args()
    # Derive repo directory from URL
    args.git_repo_dir = args.git_repo_url.split("/")[-1].replace(".git", "")
    # Construct prefix
    args.prefix = f"gcloud alpha compute tpus tpu-vm ssh {args.resource_name} --zone={args.gcp_zone} --project={args.gcp_project}"

    return args


def run_gcloud_command(command):
    try:
        result: int = subprocess.call(command, shell=True)
        assert result == 0, "Shell command fail"
        return result
    except Exception as e:
        print("Error:", str(e))


def format_gcloud_command(args, command):
    return f"{args.prefix} --command={shlex.quote(command)} --worker=all"


def pull_repo(args):
    repo_url = args.git_repo_url
    repo_name = os.path.basename(repo_url).replace(".git", "")
    cmd = (
        f"rm -rf {repo_name}; git clone {repo_url} -b {args.git_branch}; cd {repo_name}"
    )
    if args.git_commit:
        cmd += f"; git checkout {args.git_commit}"
    return format_gcloud_command(args, cmd)


def install_dependencies(args):
    command = (
        f"cd {args.git_repo_dir}; pip install -e .; pip install -r requirements_tpu.txt"
    )
    return format_gcloud_command(args, command)


def kill_python_process(args):
    command = (
        "sudo pkill -9 python; "
        "sudo lsof -w /dev/accel0 2>/dev/null | grep .py | "
        "awk '{print \"sudo kill -9 \" $2}' | sh; "
        "sudo rm -f /tmp/libtpu_lockfile"
    )
    return format_gcloud_command(args, command)


def install_conda(args):
    command = (
        'if [ ! -d "$HOME/miniconda3" ]; then '
        "    echo 'Installing conda...'; "
        "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; "
        "    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3; "
        "    rm Miniconda3-latest-Linux-x86_64.sh; "
        "else "
        "    echo 'Conda is already installed.'; "
        "fi;"
    )
    return format_gcloud_command(args, command)


def setup_environment(args):
    command = f"cd {args.git_repo_dir}; \
        source $HOME/miniconda3/bin/activate; \
        if ! conda env list | grep -q '^fsdp-jax[[:space:]]'; then \
        echo 'Creating conda environment jax-oasis from environment.yml...'; \
        CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes conda env create -y -f environment.yml -n fsdp-jax; \
        else \
        echo 'Conda environment fsdp-jax already exists.'; \
        fi; \
        conda activate fsdp-jax; \
        source $HOME/miniconda3/bin/activate fsdp-jax; \
        sudo chmod -R 777 /tmp/tpu_logs/;"
    return format_gcloud_command(args, command)


def run_command(args):
    command = f"cd {args.git_repo_dir} ; \
        sudo chmod -R 777 /tmp/tpu_logs/; \
        {args.run_command} "
    return format_gcloud_command(args, command)


if __name__ == "__main__":
    args = parse_args()

    run_gcloud_command(kill_python_process(args))
    run_gcloud_command(pull_repo(args))
    if not args.no_setup:
        run_gcloud_command(install_conda(args))
        run_gcloud_command(setup_environment(args))
        run_gcloud_command(install_dependencies(args))

    run_gcloud_command(run_command(args))
