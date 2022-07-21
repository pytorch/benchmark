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

@dataclass
class BenchmarkModelConfig:
    models: Optional[List[str]]
    device: str
    test: str
    batch_size: Optional[int]
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

def get_subrun_key(device, test, batch_size=None):
    return f"{test}-{device}-bsize_{batch_size}" if batch_size else f"{test}-{device}"

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

    bm_matrix = [devices, tests, batch_sizes]
    for device, test, batch_size in itertools.product(*bm_matrix):
        subrun = (device, test, batch_size) if batch_size else (device, test)
        out[subrun] = []
        for args in config["args"]:
            out[subrun].append(BenchmarkModelConfig(models=models, device=device, test=test, \
                               batch_size=batch_size, args=args.split(" "), \
                               rewritten_option=rewrite_option(args.split(" "))))
    return out

def run_bmconfig(config: BenchmarkModelConfig, repo_path: Path, output_path: Path, dryrun=False):
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
    subprocess.check_call(cmd, cwd=repo_path)

def gen_output_csv(output_path: Path, base_key: str):
    result = analyze_result(output_path.joinpath("json").absolute(), base_key=base_key)
    with open(output_path.joinpath("summary.csv"), "w") as sw:
        sw.write(result)

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
    for subrun in total_run:
        subrun_key = get_subrun_key(*subrun)
        bmconfigs = total_run[subrun]
        assert len(bmconfigs), f"Size of subrun {subrun} must be larger than zero."
        subrun_path = output_path.joinpath(subrun_key)
        subrun_path.mkdir(exist_ok=True, parents=True)
        for bm in bmconfigs:
            run_bmconfig(bm, repo_path, subrun_path, args.dryrun)
        if not args.dryrun:
            gen_output_csv(subrun_path, base_key=bmconfigs[0].rewritten_option)
