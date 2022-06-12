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
REPO_DIR = Path(__file__).parent.parent.parent

with add_path(REPO_DIR):
    from torchbenchmark import _list_model_paths

@dataclass
class BenchmarkModelConfig:
    models: Optional[List[str]]
    device: str
    test: str
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

def parse_bmconfigs(repo_path: Path, config_name: str) -> List[BenchmarkModelConfig]:
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    config_file = repo_path.joinpath("configs").joinpath(*config_name.split("/"))
    if not config_file.exists():
        raise RuntimeError(f"Benchmark model config {config_file} does not exist.")
    with open(config_file, "r") as cf:
        config = yaml.safe_load(cf)
    out = []
    models = get_models(config)
    for device, test, args in itertools.product(*[config["device"], config["test"], config["args"]]):
        out.append(BenchmarkModelConfig(models=models, device=device, test=test, args=args.split(" "), rewritten_option=rewrite_option(args.split(" "))))
    return out

def create_dir_if_nonexist(dirpath: str) -> Path:
    path = Path(dirpath)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path.joinpath("json")
    json_path.mkdir(parents=True, exist_ok=True)
    return path

def run_bmconfig(config: BenchmarkModelConfig, repo_path: Path, output_path: Path, dryrun=False):
    cmd = [sys.executable, "run_sweep.py", "-d", config.device, "-t", config.test]
    if config.models:
        cmd.append("-m")
        cmd.extend(config.models)
    if config.args != ['']:
        cmd.extend(config.args)
    cmd.extend(["-o", os.path.join(output_path.absolute(), "json", f"{config.rewritten_option}.json")])
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
    output_path = create_dir_if_nonexist(args.output_dir)
    bmconfig_list = parse_bmconfigs(repo_path, args.config)
    assert len(bmconfig_list), "Size of the BenchmarkModel list must be larger than zero."
    for bmconfig in bmconfig_list:
        run_bmconfig(bmconfig, repo_path, output_path, args.dryrun)
    if not args.dryrun:
        gen_output_csv(output_path, base_key=bmconfig_list[0].rewritten_option)
