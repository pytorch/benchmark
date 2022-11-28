import argparse
import time
import sys
import subprocess
from datetime import datetime

from .result_analyzer import analyze
from typing import List
from ..utils import dump_output, get_output_dir, get_output_json, add_path, REPO_PATH

with add_path(REPO_PATH):
    from utils.cuda_utils import DEFAULT_CUDA_VERSION, CUDA_VERSION_MAP

BM_NAME = "cuda-compare"

def install_nightlies(dryrun):
    default_cuda_version = CUDA_VERSION_MAP[DEFAULT_CUDA_VERSION]["pytorch_url"]
    install_cmd = ["pip", "install", "--pre", "torch", "torchvision", "torchtext", "torchaudio",
                    "-f", f"https://download.pytorch.org/whl/nightly/{default_cuda_version}/torch_nightly.html"]
    print(f"Installing pytorch packages: {install_cmd}")
    if not dryrun:
        subprocess.check_call(install_cmd, cwd=REPO_PATH)

def install_torchbench(dryrun):
    install_cmd = [sys.executable, "install.py"]
    print(f"Installing torchbench: {install_cmd}")
    if not dryrun:
        subprocess.check_call(install_cmd, cwd=REPO_PATH)

def run_benchmark(output_path, config, dryrun=False):
    benchmark_script = REPO_PATH.joinpath(".github", "scripts", "run-config.py")
    benchmark_cmd = [sys.executable, str(benchmark_script), "-c", config, "-b", str(REPO_PATH), "-o", str(output_path)]
    print(f"Running benchmark: {benchmark_cmd}")
    if not dryrun:
        subprocess.check_call(benchmark_cmd, cwd=REPO_PATH)

def dump_result_to_json(metrics):
    result = get_output_json(BM_NAME, metrics)
    dump_output(BM_NAME, result)

def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")

def get_work_dir(output_dir):
    work_dir = output_dir.joinpath(f"run-{get_timestamp()}")
    work_dir.mkdir(exist_ok=True, parents=True)
    return work_dir

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action='store_true', help="Only generate the test scripts. Do not run the benchmark.")
    parser.add_argument("--config", "-c", type=str, default="devinfra/cuda-113-116-compare", help="Specify the config file")
    parser.add_argument("--analyze", type=str, help="Only analyze the result of the specified work directory.")
    args = parser.parse_args(args)
    return args

def run(args: List[str]):
    args = parse_args(args)
    if args.analyze:
        metrics = analyze(args.analyze)
        dump_result_to_json(metrics)
        return
    work_dir = get_work_dir(get_output_dir(BM_NAME))
    install_nightlies(args.dryrun)
    install_torchbench(args.dryrun)
    run_benchmark(work_dir, args.config, dryrun=args.dryrun)
    if not args.dryrun:
        metrics = analyze(work_dir)
        dump_result_to_json(metrics)