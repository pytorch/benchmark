"""
Run PyTorch nightly benchmarking.
"""
import argparse
import dataclasses
import itertools
import pathlib
import json
import os
import shutil
import copy
import yaml
import re
import ast
import numpy

from typing import List, Dict, Optional, Union
from ..group_bench.run import (
    TorchBenchGroupBenchConfig,
    load_group_config,
    init_output_dir,
    get_metrics,
    run_config,
    run_config_accuracy,
    config_to_str,
)
from ..utils import get_output_json, get_default_output_json_path, get_default_debug_output_dir
from . import BM_NAME

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG_DIR = os.path.join(CURRENT_DIR, "configs")


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="YAML config to specify group of tests to run.")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    parser.add_argument("--debug", action="store_true", help="Save the debug output.")
    parser.add_argument("--output", default=get_default_output_json_path(BM_NAME), help="Output torchbench userbenchmark metrics file path.")
    return parser.parse_args(args)

def run(args: List[str]):
    args = parse_args(args)
    group_config: TorchBenchGroupBenchConfig = load_group_config(args.config)
    debug_output_dir = get_default_debug_output_dir(args.output) if args.debug else None
    if debug_output_dir:
        init_output_dir(group_config.configs, debug_output_dir)

    results = {}
    try:
        for config in group_config.configs:
            metrics = get_metrics(group_config.metrics, config)
            if "accuracy" in metrics:
                metrics_dict = run_config_accuracy(config, metrics, dryrun=args.dryrun)
            else:
                metrics_dict = run_config(config, metrics, dryrun=args.dryrun)
            config_str = config_to_str(config)
            for metric in metrics_dict:
                results[f"{config_str}, metric={metric}"] = metrics_dict[metric]
    except KeyboardInterrupt:
        print("User keyboard interrupted!")
    result = get_output_json(BM_NAME, results)
    if group_config.baseline_configs[0].device == 'cuda':
        import torch
        result["environ"]["device"] = torch.cuda.get_device_name()
    print(json.dumps(result, indent=4))
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)
