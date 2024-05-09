"""
Run PyTorch nightly benchmarking.
"""
import argparse
import os
import json
from typing import List
from ..utils import REPO_PATH, add_path, get_output_json, get_default_output_json_path
from . import BM_NAME

with add_path(REPO_PATH):
    from userbenchmark.group_bench.run_config import run_benchmark_config

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_DELTA_THRESHOLD = 0.07

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=os.path.join(CURRENT_DIR, "nightly.yaml"), help="YAML config to specify tests to run.")
    parser.add_argument("--run-bisect", help="Run with the output of regression detector.")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    parser.add_argument("--output", default=get_default_output_json_path(BM_NAME), help="Specify the path of the output file")
    return parser.parse_args(args)

def run(args: List[str]):
    args = parse_args(args)
    assert os.path.exists(args.config), f"Expect an existing benchmark config file, get path: {args.config}."
    benchmark_result = get_output_json(BM_NAME, run_benchmark_config(config_file=args.config, dryrun=args.dryrun))
    benchmark_result["environ"]["benchmark_style"] = "grouped"
    benchmark_result_json = json.dumps(benchmark_result, indent=4)
    with open(args.output, "w") as fp:
        fp.write(benchmark_result_json)
