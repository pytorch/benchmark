"""
Script that runs torchbench with a benchmarking config.
The configs are located within the configs/ directory.
For example, the default config we use is `torchdynamo/eager-overhead`
"""
import sys
import os
import yaml
import argparse
import subprocess
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List

from bmutils.summarize import analyze_result

@dataclass
class BenchmarkModelConfig:
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

def parse_bmconfigs(repo_path: Path, config_name: str) -> List[BenchmarkModelConfig]:
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    config_file = repo_path.joinpath("configs").joinpath(*config_name.split("/"))
    if not config_file.exists():
        raise RuntimeError(f"Benchmark model config {config_file} does not exist.")
    with open(config_file, "r") as cf:
        config = yaml.safe_load(cf)
    out = []
    for device, test, args in itertools.product(*[config["device"], config["test"], config["args"]]):
        out.append(BenchmarkModelConfig(device=device, test=test, args=args.split(" "), rewritten_option=rewrite_option(args.split(" "))))
    return out

def create_dir_if_nonexist(dirpath: str) -> Path:
    path = Path(dirpath)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path.joinpath("json")
    json_path.mkdir(parents=True, exist_ok=True)
    return path

def run_bmconfig(config: BenchmarkModelConfig, repo_path: Path, output_path: Path):
    cmd = [sys.executable, "run_sweep.py", "-d", config.device, "-t", config.test]
    if config.args != ['']:
        cmd.extend(config.args)
    cmd.extend(["-o", os.path.join(output_path.absolute(), "json", f"{config.rewritten_option}.json")])
    print(f"Now running benchmark command: {cmd}.")
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
    args = parser.parse_args()
    repo_path = Path(args.benchmark_repo)
    assert repo_path.exists(), f"Path {args.benchmark_repo} doesn't exist. Exit."
    output_path = create_dir_if_nonexist(args.output_dir)
    bmconfig_list = parse_bmconfigs(repo_path, args.config)
    assert len(bmconfig_list), "Size of the BenchmarkModel list must be larger than zero."
    for bmconfig in bmconfig_list:
        run_bmconfig(bmconfig, repo_path, output_path)
    gen_output_csv(output_path, base_key=bmconfig_list[0].rewritten_option)
