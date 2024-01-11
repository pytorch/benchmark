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

from typing import List, Dict, Optional, Any, Union
from ..utils import REPO_PATH, add_path, get_output_json, get_default_output_json_path, get_default_debug_output_dir
from . import BM_NAME

from torchbenchmark.util.experiment.instantiator import list_models, load_model_isolated, TorchBenchModelConfig, \
                                                        list_devices, list_tests
from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics, get_model_accuracy

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG_DIR = os.path.join(CURRENT_DIR, "configs")

@dataclasses.dataclass
class TorchBenchGroupBenchConfig:
    baseline_config: TorchBenchModelConfig
    metrics: List[str]
    group_configs: Dict[str, List[TorchBenchModelConfig]]

    @property
    def configs(self):
        return [ c for configs in self.group_configs.values() for c in configs ]

def config_to_str(config: TorchBenchModelConfig) -> str:
    metrics_base = f"model={config.name}, test={config.test}, device={config.device}," + \
        f" bs={config.batch_size}, extra_args={config.extra_args}"
    return metrics_base

def str_to_config(metric_name: str) -> TorchBenchModelConfig:
    regex = "model=(.*), test=(.*), device=(.*), bs=(.*), extra_args=(.*), metric=(.*)"
    model, test, device, batch_size, extra_args, _metric = re.match(regex, metric_name).groups()
    extra_args = ast.literal_eval(extra_args)
    batch_size = ast.literal_eval(batch_size)
    return TorchBenchModelConfig(
        name=model,
        test=test,
        device=device,
        batch_size=batch_size,
        extra_args=extra_args,
    )

def generate_model_configs(devices: List[str], tests: List[str], batch_sizes: List[str], model_names: List[str], extra_args: List[str]) -> List[TorchBenchModelConfig]:
    """Use the default batch size and default mode."""
    if not model_names:
        model_names = list_models()
    cfgs = itertools.product(*[devices, tests, batch_sizes, model_names])
    result = [TorchBenchModelConfig(
        name=model_name,
        device=device,
        test=test,
        batch_size=None if not batch_size else int(batch_size),
        extra_args=extra_args,
        extra_env=None,
    ) for device, test, batch_size, model_name in cfgs]
    return result

def init_output_dir(configs: List[TorchBenchModelConfig], output_dir: pathlib.Path) -> List[TorchBenchModelConfig]:
    result = []
    for config in configs:
        config_str = config_to_str(config)
        config.output_dir = output_dir.joinpath(config_str)
        if config.output_dir.exists():
            shutil.rmtree(config.output_dir)
        config.output_dir.mkdir(parents=True)
        result.append(config)
    return result

def get_metrics(metrics: List[str], config: TorchBenchModelConfig) -> List[str]:
    if metrics:
        return metrics
    if "--accuracy" in config.extra_args:
        return ["accuracy"]
    return ["latencies", "cpu_peak_mem", "gpu_peak_mem"]

def validate(candidates: List[str], choices: List[str]) -> List[str]:
    """Validate the candidates provided by the user is valid"""
    for candidate in candidates:
        assert candidate in choices, f"Specified {candidate}, but not in available list: {choices}."
    return candidates

def parse_str_to_list(candidates: Optional[str]) -> List[str]:
    if isinstance(candidates, list):
        return candidates
    elif candidates == None:
        return [""]
    candidates = list(map(lambda x: x.strip(), candidates.split(",")))
    return candidates

def metrics_to_dict(metrics: Union[TorchBenchModelMetrics, Dict[str, str]]) -> Dict[str, Union[str, float]]:
    if isinstance(metrics, TorchBenchModelMetrics):
        pass
    return metrics

def run_config(config: TorchBenchModelConfig, metrics: List[str], dryrun: bool=False) -> Dict[str, Union[str, float]]:
    """This function only handles NotImplementedError, all other errors will fail."""
    print(f"Running {config} ...", end='', flush=True)
    if dryrun:
        print(" [skip_by_dryrun]", flush=True)
        return dict.fromkeys(metrics, "skip_by_dryrun")
    # We do not allow RuntimeError in this test
    try:
        # load the model instance in subprocess
        model = load_model_isolated(config)
        # get the model test metrics
        metrics_output: TorchBenchModelMetrics = get_model_test_metrics(model, metrics=metrics)
        result = {}
        for metric in metrics:
            if metric == "latencies" and metrics_output.latencies:
                result[metric] = numpy.median(metrics_output.latencies)
            else:
                result[metric] = getattr(metrics_output, metric, None)
                result[metric] = "failed" if result[metric] == None else result[metric]
        print(" [done]", flush=True)
        return result
    except NotImplementedError:
        print(" [not_implemented]", flush=True)
        return dict.fromkeys(metrics, "not_implemented")
    except OSError as e:
        print(" [oserror]", flush=True)
        return dict.fromkeys(metrics, str(e))

def run_config_accuracy(config: TorchBenchModelConfig, metrics: List[str], dryrun: bool=False) -> Dict[str, str]:
    assert metrics == ["accuracy"], f"When running accuracy test, others metrics are not supported: {metrics}."
    print(f"Running {config} ...", end='', flush=True)
    if dryrun:
        print(" [skip_by_dryrun]", flush=True)
        return {"accuracy": "skip_by_dryrun"}
    try:
        accuracy = get_model_accuracy(config)
        print(" [done]", flush=True)
        return {"accuracy": accuracy}
    except NotImplementedError:
        print(" [not_implemented]", flush=True)
        return {"accuracy": "not_implemented"}
    except OSError as e:
        print(" [oserror]", flush=True)
        return {"accuracy": str(e)}

def load_group_config(config_file: str) -> TorchBenchGroupBenchConfig:
    if not os.path.exists(config_file):
        config_file = os.path.join(DEFAULT_CONFIG_DIR, config_file)
    with open(config_file, "r") as fp:
        data = yaml.safe_load(fp)
    baseline_config = TorchBenchModelConfig(
        name=data["model"],
        test=data["test"],
        device=data["device"],
        batch_size=data["batch_size"] if "batch_size" in data else None,
        extra_args=data["extra_args"].split(" ") if "extra_args" in data else [],
    )
    metrics = data["metrics"] if "metrics" in data else []
    group_configs = {}
    for group_name in data["test_group"]:
        group_configs[group_name] = []
        group_extra_args = data["test_group"][group_name]["extra_args"].split(" ")
        subgroup_extra_args_list = list(map(lambda x: x["extra_args"].split(" "), data["test_group"][group_name]["subgroup"]))
        for subgroup_extra_args in subgroup_extra_args_list:
            subgroup_config = copy.deepcopy(baseline_config)
            subgroup_config.extra_args.extend(group_extra_args)
            subgroup_config.extra_args.extend(subgroup_extra_args)
            group_configs[group_name].append(subgroup_config)
    return TorchBenchGroupBenchConfig(baseline_config, metrics, group_configs)

def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="YAML config to specify group of tests to run.")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    parser.add_argument("--debug", action="store_true", help="Save the debug output.")
    parser.add_argument("--output", default=f"/tmp/{BM_NAME}", help="Output torchbench userbenchmark metrics file path.")
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
    if group_config.baseline_config.device == 'cuda':
        import torch
        result["environ"]["device"] = torch.cuda.get_device_name()
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)
    print(json.dumps(result))
