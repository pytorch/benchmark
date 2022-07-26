import argparse
import os
import yaml
import itertools
from pathlib import Path
from typing import List
from ..utils import get_output_dir

BM_NAME = "release-test"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

RUN_SCRIPT_TEMPLATE = """
set -x
conda uninstall -y pytorch torchvision torchtext ${MAGMA_VERSION}
. switch-cuda.sh {CUDA_VERSION}
# install magma and cudatoolkit
conda install -y cudatoolkit=${CUDA_VERSION}
conda install -y -c pytorch ${MAGMA_VERSION}
# install pytorch
conda install -y -c {PYTORCH_CHANNEL} pytorch={PYTORCH_VERSION} torchvision torchtext
python -c 'import torch; print(torch.__version__); print(torch.version.git_version)'
sudo nvidia-smi -ac ${GPU_FREQUENCY}
# check machine tuned
pip install -U py-cpuinfo psutil distro
python ../torchbenchmark/util/machine_config.py
# run benchmarks
"""

def generate_test_scripts(config, log_dir):
    assert "cuda" in config and isinstance(config["cuda"], list), f"Expected CUDA config list, but not found."
    assert "pytorch" in config and isinstance(config["pytorch"], list), f"Exptected pytorch version list, but not found."
    bm_matrix = [config["cuda"], config["pytorch"]]
    for cuda_ver, pytorch_ver in itertools.product(*bm_matrix):
        run_script = ""

def get_log_dir(output_dir, config_name):
    log_dir = output_dir.joinpath("logs", config_name)
    log_dir.mkdir(exist_ok=True, parents=True)
    return log_dir

def get_config(config_name: str):
    if os.path.exists(os.path.join(DEFAULT_CONFIG_PATH, config_name)):
        config_name = os.path.join(DEFAULT_CONFIG_PATH, config_name)
    elif os.path.exists(os.path.join(DEFAULT_CONFIG_PATH, config_name, ".yaml")):
        config_name = os.path.join(DEFAULT_CONFIG_PATH, config_name, ".yaml")
    else:
        raise ValueError(f"Can't find config name {config_name} in config path {DEFAULT_CONFIG_PATH}.")
    with open(config_name, "r") as yfile:
        config = yaml.safe_load(yfile)
    return config

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config for release testing")
    args = parser.parse_args(args)
    return args

def prepare_release_tests(args: argparse.Namespace, output_dir: Path):
    config = get_config(args.config)
    log_dir = get_log_dir(output_dir, args.config)
    run_scripts = generate_test_scripts(config, log_dir)

def run(args: List[str]):
    args = parse_args(args)
    prepare_release_tests(extra_args=args, output_dir=get_output_dir())
