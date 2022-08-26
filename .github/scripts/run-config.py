"""
Script that runs torchbench with a benchmarking config.
The configs are located within the configs/ directory.
For example, the default config we use is `torchdynamo/eager-overhead`
"""
import re
import sys
import os
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

CUDA_VERSION_MAP = {
    "11.3": {
        "pytorch_url": "cu113",
        "magma_version": "magma-cuda113",
    },
    "11.6": {
         "pytorch_url": "cu116",
         "magma_version": "magma-cuda116",
    },
    "11.7": {
         "pytorch_url": "cu117",
         "magma_version": "magma-cuda117",
    }
}

@dataclass
class BenchmarkModelConfig:
    models: Optional[List[str]]
    device: str
    test: str
    batch_size: Optional[int]
    cuda_version: Optional[str]
    args: List[str]
    rewritten_option: str

def rewrite_option(option: List[str]) -> str:
    out = []
    for x in option:
        out.append(x.replace("--", ""))
    if option == ['']:
        return "eager"
    else:
        return "-".join(out)

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

def get_subrun_key(subrun_key):
    return "-".join(subrun_key)

def get_cuda_versions(config):
    if not "cuda_version" in config:
        return [None]
    return config["cuda_version"]

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

def get_subrun(device, test, batch_size, cuda_version):
    if not batch_size and not cuda_version:
        return (device, test)
    if not batch_size:
        return (device, test, f"cuda_{cuda_version}")
    if not cuda_version:
        return (device, test, f"bs_{batch_size}")
    return (device, test, f"bs_{batch_size}", f"cuda_{cuda_version}")

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

    bm_matrix = [devices, tests, batch_sizes, cuda_versions]
    for device, test, batch_size, cuda_version in itertools.product(*bm_matrix):
        subrun = get_subrun(device, test, batch_size, cuda_version)
        out[subrun] = []
        for args in config["args"]:
            out[subrun].append(BenchmarkModelConfig(models=models, device=device, test=test, \
                               batch_size=batch_size, cuda_version=cuda_version, args=args.split(" "), \
                               rewritten_option=rewrite_option(args.split(" "))))
    return out

def prepare_bmconfig_env(config: BenchmarkModelConfig, repo_path: Path, dryrun=False):
    """Prepare the correct cuda version environment for the benchmarking."""
    if not config.cuda_version:
        return
    cuda_version = config.cuda_version
    # step 1: setup CUDA path and environment variables
    env = os.environ
    cuda_path = Path("/").joinpath("usr", "local", f"cuda-{cuda_version}")
    assert cuda_path.exists() and cuda_path.is_dir(), f"Expected CUDA Library path {cuda_path} doesn't exist."
    env["CUDA_ROOT"] = str(cuda_path)
    env["CUDA_HOME"] = str(cuda_path)
    env["PATH"] = f"{str(cuda_path)}/bin:{env['PATH']}"
    env["LD_LIBRARY_PATH"] = f"{str(cuda_path)}/lib64:{str(cuda_path)}/extras/CUPTI/lib64:{env['LD_LIBRARY_PATH']}"
    # step 2: test call nvcc to confirm the version
    test_nvcc = ["nvcc", "--version"]
    if not dryrun:
        subprocess.check_call(test_nvcc)
    # step 1: uninstall all pytorch packages
    uninstall_torch_cmd = ["pip", "uninstall", "-y", "torch", "torchvision", "torchtext"]
    print(f"Uninstall pytorch: {uninstall_torch_cmd}")
    if not dryrun:
        for _loop in range(3):
            subprocess.check_call(uninstall_torch_cmd)
    # step 2: install pytorch nightly with the correct cuda version
    install_magma_cmd = ["conda", "install", "-c", "pytorch", CUDA_VERSION_MAP[cuda_version]['magma_version']]
    print(f"Install magma: {install_magma_cmd}")
    if not dryrun:
        subprocess.check_call(install_magma_cmd)
    pytorch_nightly_url = f"https://download.pytorch.org/whl/nightly/{CUDA_VERSION_MAP[cuda_version]['pytorch_url']}/torch_nightly.html"
    install_torch_cmd = ["pip", "install", "--pre", "torch", "torchvision", "torchtext", "-f",  pytorch_nightly_url]
    print(f"Install pytorch nightly: {install_torch_cmd}")
    if not dryrun:
        subprocess.check_call(install_torch_cmd)
    # step 3: install torchbench
    install_torchbench_cmd = [sys.executable, "install.py"]
    print(f"Install torchbench: {install_torchbench_cmd}")
    if not dryrun:
        subprocess.check_call(install_torchbench_cmd, cwd=repo_path)
    return env

def run_bmconfig(config: BenchmarkModelConfig, repo_path: Path, output_path: Path, dryrun=False):
    run_env = prepare_bmconfig_env(config, repo_path=repo_path, dryrun=dryrun)
    cmd = [sys.executable, "run_sweep.py", "-d", config.device, "-t", config.test]
    if config.batch_size:
        cmd.append("-b")
        cmd.append(str(config.batch_size))
    if config.models:
        cmd.append("-m")
        cmd.extend(config.models)
    if config.args != ['']:
        cmd.extend(config.args)
    output_dir = output_path.joinpath("json")
    output_dir.mkdir(exist_ok=True, parents=True)
    cmd.extend(["-o", os.path.join(output_dir.absolute(), f"{config.rewritten_option}.json")])
    print(f"Now running benchmark command: {cmd}.", flush=True)
    if dryrun:
        return
    subprocess.check_call(cmd, cwd=repo_path, env=run_env)

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
            run_bmconfig(bm, repo_path, subrun_path, args.dryrun)
        if not args.dryrun:
            gen_output_csv(subrun_path, base_key=bmconfigs[0].rewritten_option)
