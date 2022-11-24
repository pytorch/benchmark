"""
Script that runs torchbench with a benchmarking config.
The configs are located within the configs/ directory.
For example, the default config we use is `torchdynamo/eager-overhead`
"""
import re
import sys
import os
from charset_normalizer import logging
import yaml
import argparse
import subprocess
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from bmutils import add_path
from bmutils.summarize import analyze_result

REPO_DIR = str(Path(__file__).parent.parent.parent.resolve())

with add_path(REPO_DIR):
    from torchbenchmark import _list_model_paths
    from utils.cuda_utils import prepare_cuda_env, install_pytorch_nightly

NAME_MAP = {
    "backend-blade": "disc",
    "backend-blade-trt": "blade",
    "backend-torchscript": "ofi",
    "fuser-fuser1-jit": "nnc",
    "fuser-fuser2-jit": "nvfuser",
    "torchdynamo-eager": "dynamo-eager",
    "torchdynamo-nvfuser" : "dynamo-nvfuser",
    "torchdynamo-blade_optimize_dynamo": "dynamo-disc",
    "torchdynamo-blade_optimize_dynamo-trt": "dynamo-blade",
    "torchdynamo-cudagraphs": "dynamo-cudagraphs",
    "torchdynamo-onnxrt_cpu": "dynamo-onnxrt_cpu",
    "torchdynamo-ipex": "dynamo-ipex",
    "torchdynamo-ofi": "dynamo-ofi"
}
@dataclass
class BenchmarkModelConfig:
    models: Optional[List[str]]
    device: str
    test: str
    batch_size: Optional[int]
    cuda_version: Optional[str]
    precision: Optional[str]
    args: List[str]
    rewritten_option: str

def rewrite_option(option: List[str]) -> str:
    out = []
    for x in option:
        out.append(x.replace("--", ""))
    if option == ['']:
        return "eager"
    else:
        return NAME_MAP["-".join(out)]

def get_models(config) -> Optional[str]:
    # if the config doesn't specify the 'models' key,
    # returns None (means running all models)
    if not "models" in config:
        return None
    # get list of models
    models = list(map(lambda x: os.path.basename(x), _list_model_paths()))
    enabled_models = []
    for model_pattern in config["models"]:
        r = re.compile(model_pattern)
        matched_models = list(filter(lambda x: r.match(x), models))
        enabled_models.extend(matched_models)
    assert enabled_models, f"The model patterns you specified {config['models']} does not match any model. Please double check."
    return enabled_models

def get_cuda_versions(config):
    if not "cuda_version" in config:
        return [None]
    return config["cuda_version"]

def get_subrun_key(subrun_key):
    return "-".join(subrun_key)

def get_tests(config):
    if not "test" in config:
        return ["train", "eval"]
    return config["test"]

def get_devices(config):
    if not "device" in config:
        return ["cpu", "cuda"]
    return config["device"]

def get_batch_sizes(config):
    if not "batch_size" in config:
        return [None]
    return config["batch_size"]

def get_precisions(config):
    if not "precision" in config:
        return [""]
    return config["precision"]

def get_subrun(device, test, batch_size, cuda_version, precision):
    subrun = [test, device]
    if batch_size:
        subrun.append(f"bs_{batch_size}")
    if cuda_version:
        subrun.append(f"cuda_{cuda_version}")
    if precision:
        subrun.append(precision)
    return tuple(subrun)

def parse_bmconfigs(repo_path: Path, config_name: str) -> List[BenchmarkModelConfig]:
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    config_file = repo_path.joinpath("configs").joinpath(*config_name.split("/"))
    if not config_file.exists():
        raise RuntimeError(f"Benchmark model config {config_file} does not exist.")
    with open(config_file, "r") as cf:
        config = yaml.safe_load(cf)
    out = {}

    models = get_models(config)
    devices = get_devices(config)
    tests = get_tests(config)
    batch_sizes = get_batch_sizes(config)
    cuda_versions = get_cuda_versions(config)
    precisions = get_precisions(config)

    bm_matrix = [devices, tests, batch_sizes, cuda_versions, precisions]
    for device, test, batch_size, cuda_version, precision in itertools.product(*bm_matrix):
        subrun = get_subrun(device, test, batch_size, cuda_version, precision)
        out[subrun] = []
        for args in config["args"]:
            out[subrun].append(BenchmarkModelConfig(models=models, device=device, test=test, \
                               batch_size=batch_size, cuda_version=cuda_version, precision=precision, \
                               args=args.split(" "), rewritten_option=rewrite_option(args.split(" "))))

    return out

def prepare_bmconfig_env(config: BenchmarkModelConfig, repo_path: Path, dryrun=False):
    """Prepare the correct cuda version environment for the benchmarking."""
    if not config.cuda_version:
        return os.environ.copy()
    cuda_version = config.cuda_version
    new_env = prepare_cuda_env(cuda_version=cuda_version)
    install_pytorch_nightly(cuda_version=cuda_version, env=new_env, dryrun=dryrun)
    return new_env

def run_bmconfig(config: BenchmarkModelConfig, repo_path: Path, output_path: Path, dryrun=False):
    run_env = prepare_bmconfig_env(config, repo_path=repo_path, dryrun=dryrun)
    cmd = [sys.executable, "run_sweep.py", "-d", config.device, "-t", config.test]
    if config.batch_size:
        cmd.append("-b")
        cmd.append(str(config.batch_size))
    if config.models:
        cmd.append("-m")
        cmd.extend(config.models)
    if config.precision:
        cmd.append("--precision")
        cmd.append(config.precision)
    if config.args != ['']:
        cmd.extend(config.args)
    output_dir = output_path.joinpath("json")
    output_dir.mkdir(exist_ok=True, parents=True)
    cmd.extend(["-o", os.path.join(output_dir.absolute(), f"{config.rewritten_option}.json")])
    print(f"Now running benchmark command: {cmd}.", flush=True)
    if dryrun:
        return
    subprocess.check_call(cmd, cwd=repo_path, env=run_env)

def run_bmconfig_profiling(config: BenchmarkModelConfig, repo_path: Path, output_path: Path, dryrun=False):
    nsys_path_cmd = ["which", "nsys"]
    nsys_path = subprocess.run(nsys_path_cmd, stdout=subprocess.PIPE).stdout
    if not nsys_path:
        logging.error("nsys not found in PATH, profiling script not work!" \
            "Nsys install guidelines can be found in https://developer.nvidia.com/blog/nvidia-nsight-systems-containers-cloud/")
        return

    run_sweep_cmd = [sys.executable, "run_sweep.py", "-d", config.device, "-t", config.test, "--is-profiling"]
    if config.batch_size:
        run_sweep_cmd.append("-b")
        run_sweep_cmd.append(str(config.batch_size))
    if config.precision:
        run_sweep_cmd.append("--precision")
        run_sweep_cmd.append(config.precision)
    if config.args != ['']:
        run_sweep_cmd.extend(config.args)
    
    # mkdir for profiling output
    output_dir = output_path.joinpath("profiling")
    output_dir.mkdir(exist_ok=True, parents=True)
    run_sweep_cmd.append("-m")

    # list profiling models
    models = config.models or [os.path.basename(model_path) for model_path in _list_model_paths()]
    for model in models:
        run_sweep_cmd.append(model)
        model_profiling_dir = output_dir.joinpath(model).absolute()
        model_profiling_dir.mkdir(exist_ok=True, parents=True)
        model_prefix = os.path.join(model_profiling_dir, f"{config.rewritten_option}")

        # profiling cmd
        profiling_cmd = ["nsys", "profile", "-f", "true", "--wait=primary", "-c", "cudaProfilerApi", "-o", model_prefix]

        # stats command
        stats_cmd = ["nsys", "stats", "--report", "gputrace", "-q", "-f", "csv", "-o", model_prefix, model_prefix + ".nsys-rep"]
        # use parse script to gen gputrace.csv
        parse_cmd = [sys.executable, "parse_nsys_result.py", model_prefix + "_gputrace.csv"]
        try:
            print(f"Now profiling benchmark command: {profiling_cmd + run_sweep_cmd}.", flush=True)
            subprocess.run(profiling_cmd + run_sweep_cmd, cwd=repo_path)
            print(f"Now stats benchmark command: {stats_cmd}.", flush=True)
            subprocess.check_call(stats_cmd, cwd=repo_path)
            print(f"Now parse benchmark command: {parse_cmd}.", flush=True)
            with open(model_prefix + ".csv", "w") as fd:
                subprocess.check_call(parse_cmd, cwd=repo_path, stdout=fd)
        except subprocess.CalledProcessError:
            pass

        run_sweep_cmd.pop()
 

def gen_output_csv(output_path: Path, base_key: str):
    result = analyze_result(output_path.joinpath("json").absolute(), base_key=base_key)
    with open(output_path.joinpath("summary.csv"), "w") as sw:
        sw.write(result)

def check_env(bmconfigs):
    """Check that the machine has been properly setup to run the config."""
    for subrun in total_run:
        bmconfigs = total_run[subrun]
        for bmconfig in bmconfigs:
            if bmconfig.cuda_version:
                cuda_path = Path("/").joinpath("usr", "local", f"cuda-{bmconfig.cuda_version}")
                assert cuda_path.exists() and cuda_path.is_dir(), f"Expected CUDA path {str(cuda_path)} doesn't exist. Please report a bug."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Specify benchmark config to run.")
    parser.add_argument("--benchmark-repo", "-b", required=True, help="Specify the pytorch/benchmark repository location.")
    parser.add_argument("--output-dir", "-o", required=True, help="Specify the directory to save the outputs.")
    parser.add_argument("--dryrun", action="store_true", help="Dry run the script and don't run the benchmark.")
    args = parser.parse_args()
    repo_path = Path(args.benchmark_repo)
    assert repo_path.exists(), f"Path {args.benchmark_repo} doesn't exist. Exit."
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    total_run = parse_bmconfigs(repo_path, args.config)
    assert len(total_run), "Size of the BenchmarkModel list must be larger than zero."
    check_env(total_run)
    for subrun in total_run:
        subrun_key = get_subrun_key(subrun)
        bmconfigs = total_run[subrun]
        assert len(bmconfigs), f"Size of subrun {subrun} must be larger than zero."
        subrun_path = output_path.joinpath(subrun_key)
        subrun_path.mkdir(exist_ok=True, parents=True)
        for bm in bmconfigs:
            # could not together because profiling results is just one model
            run_bmconfig(bm, repo_path, subrun_path, args.dryrun)
            if "cuda" in subrun:
                run_bmconfig_profiling(bm, repo_path, subrun_path, args.dryrun)
        if not args.dryrun:
            gen_output_csv(subrun_path, base_key=bmconfigs[0].rewritten_option)
