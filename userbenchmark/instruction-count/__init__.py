import argparse
from datetime import datetime
import git
import numpy
import os
import json
import subprocess
import sys
import time
import shutil
from pathlib import Path
from ..utils import dump_output, get_output_dir, get_output_json, REPO_PATH

from typing import List

BM_NAME = "instruction-count"
RESULT_JSON = "ubenchmark_results.json"
PYTORCH_SRC_URL = "https://github.com/pytorch/pytorch.git"

def translate_result_metrics(json_path: Path):
    metrics = {}
    with open(json_path, "r") as j:
        raw_result = json.load(j)
    raw_values = raw_result["values"]
    for key in raw_values:
        median_time = numpy.median(raw_values[key]["times"])
        metrics[key] = median_time
    return metrics

def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")

def get_work_dir(output_dir):
    work_dir = output_dir.joinpath(f"run-{get_timestamp()}")
    work_dir.mkdir(exist_ok=True, parents=True)
    return work_dir

def checkout_pytorch_repo(pytorch_repo: str, pytorch_branch: str):
    git.Repo.clone_from(PYTORCH_SRC_URL, pytorch_repo, depth=1, branch=pytorch_branch)

def cleanup_pytorch_repo(pytorch_repo: str):
    pytorch_repo_path = Path(pytorch_repo)
    if pytorch_repo_path.exists():
        shutil.rmtree(pytorch_repo_path)

def run_benchmark(pytorch_src_path: Path, output_json_path: Path):
    benchmark_path = pytorch_src_path.joinpath("benchmarks", "instruction_counts")
    command = [sys.executable, "main.py", "--mode", "ci", "--destination", str(output_json_path.resolve())]
    subprocess.check_call(command, cwd=benchmark_path)

def parse_args(args: List[str], work_dir: Path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-src", default=str(work_dir.resolve()),
                        help="Location of PyTorch source repo")
    parser.add_argument("--pytorch-branch", default="master",
                        help="The branch of pytorch to check out")
    parser.add_argument("--analyze-json", type=str, default=None, help="Only analyze an existing result")
    args = parser.parse_args(args)
    return args

def run(args: List[str]):
    output_dir = get_output_dir(BM_NAME)
    work_dir = get_work_dir(output_dir)
    args = parse_args(args, work_dir)
    if args.analyze_json:
        json_path = Path(args.analyze_json)
        metrics = translate_result_metrics(json_path)
        result = get_output_json(BM_NAME, metrics)
        dump_output(BM_NAME, result)
        return
    cleanup_pytorch_repo(args.pytorch_src)
    checkout_pytorch_repo(args.pytorch_src, args.pytorch_branch)
    pytorch_src_path = Path(args.pytorch_src)
    output_json_path = work_dir.joinpath(RESULT_JSON)
    run_benchmark(pytorch_src_path, output_json_path)
    metrics = translate_result_metrics(output_json_path)
    result = get_output_json(BM_NAME, metrics)
    dump_output(BM_NAME, result)
    cleanup_pytorch_repo(args.pytorch_src)
