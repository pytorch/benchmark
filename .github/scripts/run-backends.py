"""
Script that runs different backends and generate comparison report.
Format of option list: separated by semicolon(;).
For example, if you are testing two backends, 1) torchdynamo fx2trt; 2) torchscript, specify the option as:
--torchdynamo fx2trt; --jit
We use semicolon as separator because it won't normally be used in command line options
as a special charactor in bash script.
"""
import argparse
import subprocess
from pathlib import Path
from typing import List

def parse_options(options: str) -> List[List[str]]:
    return list(map(lambda x: x.strip().split(" "), options.strip().split(";")))

def create_dir_if_nonexist(dirpath: str) -> Path:
    path = Path(dirpath)
    path.mkdir(parents=True, exist_ok=True)
    return path

def run_option(option: List[str], repo_path: Path, output_path: Path):
    print(f"Now running backend option {option}, saving result to directory {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-options", "-r", required=True, help="Specify the list of options to run the benchmark.")
    parser.add_argument("--benchmark-repo", "-b", required=True, help="Specify the pytorch/benchmark repository.")
    parser.add_argument("--output-dir", "-o", required=True, help="Specify the directory to save the outputs.")
    args = parser.parse_args()
    repo_path = Path(args.benchmark_repo)
    assert repo_path.exists(), f"Path {args.benchmark_repo} doesn't exist. Exit."
    output_path = create_dir_if_nonexist(args.output_dir)
    for option in parse_options(args.run_options):
        run_option(option, repo_path, output_path)
