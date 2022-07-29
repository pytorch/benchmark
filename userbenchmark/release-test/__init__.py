import argparse
import os
import yaml
import time
import itertools
import subprocess
from datetime import datetime
from git import Repo
from pathlib import Path
from typing import List
from ..utils import get_output_dir
from .result_analyzer import analyze

# Expected WORK_DIR structure
# WORK_DIR/
#  |---examples/
#  |---pytorch-<ver1>-cuda<ver1>/
#        |---run.sh
#        |---mnist/
#        |---mnist-hogwild/
#        |---<other-benchmarks>
#  |---pytorch-<ver2>-cuda<ver2>/
#  |---summary.csv

BM_NAME = "release-test"
EXAMPLE_URL = "https://github.com/pytorch/examples.git"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
RUN_TEMPLATE = """
bash {RELEASE_TEST_ROOT}/setup_env.sh '{CUDA_VERSION}' '{MAGMA_VERSION}' '{PYTORCH_VERSION}' '{PYTORCH_CHANNEL}' '{WORK_DIR}'
bash {RELEASE_TEST_ROOT}/run_release_test.sh '{CUDA_VERSION}' '{RESULT_DIR}'
"""

def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")

def get_work_dir(output_dir):
    work_dir = output_dir.joinpath(f"run-{get_timestamp()}")
    work_dir.mkdir(exist_ok=True, parents=True)
    return work_dir

def generate_test_scripts(config, work_dir):
    assert "cuda" in config and isinstance(config["cuda"], list), f"Expected CUDA config list, but not found."
    assert "pytorch" in config and isinstance(config["pytorch"], list), f"Exptected pytorch version list, but not found."
    bm_matrix = [config["cuda"], config["pytorch"]]
    run_scripts = {}
    for cuda, pytorch in itertools.product(*bm_matrix):
        run_key = f"pytorch-{pytorch['version']}-cuda-{cuda['version']}"
        run_script = RUN_TEMPLATE.format(RELEASE_TEST_ROOT=CURRENT_DIR,
                                         CUDA_VERSION=cuda["version"],
                                         MAGMA_VERSION=cuda["magma_version"],
                                         PYTORCH_VERSION=pytorch["version"],
                                         PYTORCH_CHANNEL=pytorch["conda_channel"],
                                         WORK_DIR=work_dir,
                                         RESULT_DIR=work_dir.joinpath(run_key))
        run_scripts[run_key] = run_script
    return run_scripts

def dump_test_scripts(run_scripts, work_dir):
    for run_key, run_script in run_scripts.items():
        run_script_loc = work_dir.joinpath(run_key)
        run_script_loc.mkdir(exist_ok=True)
        with open(run_script_loc.joinpath("run.sh"), "w") as rs:
            rs.write(run_script)

def run_benchmark(run_scripts, work_dir):
    for run_key, _rscript in run_scripts.items():
        run_script_path = work_dir.joinpath(run_key, "run.py")
        # run the benchmark
        print(f"Running benchmark {run_key} ...")
        subprocess.check_call(["bash", run_script_path])

def get_config(config_name: str):
    if os.path.exists(os.path.join(DEFAULT_CONFIG_PATH, config_name)):
        config_name = os.path.join(DEFAULT_CONFIG_PATH, config_name)
    elif os.path.exists(os.path.join(DEFAULT_CONFIG_PATH, f"{config_name}.yaml")):
        config_name = os.path.join(DEFAULT_CONFIG_PATH, f"{config_name}.yaml")
    else:
        raise ValueError(f"Can't find config name {config_name} in config path {DEFAULT_CONFIG_PATH}.")
    with open(config_name, "r") as yfile:
        config = yaml.safe_load(yfile)
    return config

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, type=str, help="Config for release testing")
    parser.add_argument("--dry-run", action='store_true', help="Only generate the test scripts. Do not run the benchmark.")
    parser.add_argument("--analyze-result", type=str, help="Only analyze the result of the specified work directory.")
    args = parser.parse_args(args)
    return args

def prepare_release_tests(args: argparse.Namespace, work_dir: Path):
    config = get_config(args.config)
    run_scripts = generate_test_scripts(config, work_dir)
    dump_test_scripts(run_scripts, work_dir)
    # clone the examples repo
    Repo.clone_from(EXAMPLE_URL, work_dir.joinpath("examples"))
    return run_scripts

def run(args: List[str]):
    args = parse_args(args)
    if args.analyze_result:
        analyze(args.analyze_result)
        return
    work_dir = get_work_dir(get_output_dir(BM_NAME))
    run_scripts = prepare_release_tests(args=args, work_dir=work_dir)
    if not args.dry_run:
        run_benchmark(run_scripts, work_dir)
        analyze(work_dir)
