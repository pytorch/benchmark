"""
Run grouped userbenchmarks.
"""
import argparse
import ast
import copy
import dataclasses
import itertools
import json
import os
import pathlib
import re
import shutil

from typing import Any, Dict, List, Optional, Union

import numpy
import yaml

from torchbenchmark.util.experiment.metrics import (
    get_model_accuracy,
    get_model_test_metrics,
    TorchBenchModelMetrics,
)

from ..task import TBUserbenchmarkConfig, TBUserTask
from userbenchmark.task import TBUserTask

from ..utils import (
    add_path,
    get_default_debug_output_dir,
    get_default_output_json_path,
    get_output_json,
    REPO_PATH,
)
from . import BM_NAME

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG_DIR = os.path.join(CURRENT_DIR, "configs")


@dataclasses.dataclass
class TBUserbenchmarkGroupConfig:
    group_configs: Dict[str, List[TBUserbenchmarkConfig]]

    @property
    def configs(self):
        return [ c for configs in self.group_configs.values() for c in configs ]


def init_output_dir(group_config: TBUserbenchmarkGroupConfig, output_dir: pathlib.Path):
    for group_name in group_config.group_configs:
        configs = group_config.group_configs[group_name]
        for config in configs:
            config_str = config.output_dir_name
            config.output_dir = output_dir.joinpath(group_name, config_str)
            if config.output_dir.exists():
                shutil.rmtree(config.output_dir)
            config.output_dir.mkdir(parents=True)


def run_config(config: TBUserbenchmarkConfig, dryrun: bool=False) -> None:
    print(f"Running {config} ...", end='', flush=True)
    if dryrun:
        print(" [skip_by_dryrun]", flush=True)
        return
    # We do not allow RuntimeError in this test
    try:
        # load the userbenchmark and run it
        task = TBUserTask(config)
        task.run(config.args)
        print(" [done]", flush=True)
        return
    except NotImplementedError:
        print(" [not_implemented]", flush=True)
        return
    except OSError:
        print(" [oserror]", flush=True)
        return

def load_group_config(config_file: str) -> TBUserbenchmarkGroupConfig:
    if not os.path.exists(config_file):
        config_file = os.path.join(DEFAULT_CONFIG_DIR, config_file)
    with open(config_file, "r") as fp:
        data = yaml.safe_load(fp)
    baseline_config = TBUserbenchmarkConfig(
        name=data["name"],
        args=data["base_args"].split(" "),
    )
    group_configs = {}
    for group_name in data["test_group"]:
        group_configs[group_name] = []
        group_extra_args = list(filter(lambda x: x, data["test_group"][group_name].get("extra_args", "").split(" ")))
        subgroup_extra_args_list = list(map(lambda x: x["extra_args"].split(" "), data["test_group"][group_name]["subgroup"]))
        for subgroup_extra_args in subgroup_extra_args_list:
            subgroup_config = copy.deepcopy(baseline_config)
            subgroup_config.args.extend(group_extra_args)
            subgroup_config.args.extend(subgroup_extra_args)
            group_configs[group_name].append(subgroup_config)
    return TBUserbenchmarkGroupConfig(group_configs)

def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="YAML config to specify group of tests to run.")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    parser.add_argument("--output", default=get_default_output_json_path(BM_NAME), help="Output torchbench userbenchmark metrics file path.")
    return parser.parse_args(args)

def run(args: List[str]):
    args = parse_args(args)
    group_config: TBUserbenchmarkGroupConfig = load_group_config(args.config)
    output_dir = get_default_debug_output_dir(args.output)
    init_output_dir(group_config, output_dir)

    try:
        for config in group_config.configs:
            run_config(config, dryrun=args.dryrun)
    except KeyboardInterrupt:
        print("User keyboard interrupted!")

    print(f"Benchmark results are saved to the output dir: {output_dir}")
