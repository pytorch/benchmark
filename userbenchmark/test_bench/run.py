"""
Run PyTorch nightly benchmarking.
"""
import argparse
import itertools
import pathlib
import json
import os
import shutil
import numpy

from typing import List, Dict, Optional, Any, Union
from ..utils import REPO_PATH, add_path, get_output_json, get_default_output_json_path, get_default_debug_output_dir
from . import BM_NAME

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, load_model_isolated, TorchBenchModelConfig, \
                                                            list_devices, list_tests
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics, get_model_accuracy

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def config_to_str(config: TorchBenchModelConfig) -> str:
    metrics_base = f"model={config.name}, test={config.test}, device={config.device}," + \
        f" bs={config.batch_size}, extra_args={config.extra_args}"
    return metrics_base

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

def get_metrics(config: TorchBenchModelConfig) -> List[str]:
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
            if metric == "latency" and metrics_output.latencies:
                result[metric] = numpy.median(metrics_output.latencies)
            if not result[metric]:
                result[metric] = "failed"
        print(" [done]", flush=True)
        return result
    except NotImplementedError as e:
        print(" [not_implemented]", flush=True)
        return dict.fromkeys(metrics, "not_implemented")    

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

def parse_known_args(args):
    parser = argparse.ArgumentParser()
    default_device = "cuda" if "cuda" in list_devices() else "cpu"
    parser.add_argument(
        "models",
        help="Name of models to run, split by comma.",
    )
    parser.add_argument("--device", "-d", default=default_device, help="Devices to run, splited by comma.")
    parser.add_argument("--test", "-t", default="eval", help="Tests to run, splited by comma.")
    parser.add_argument("--bs", default=None, help="Optionally, specify the batch size.")
    parser.add_argument("--config", "-c", default=None, help="YAML config to specify tests to run.")
    parser.add_argument("--run-bisect", help="Run with the output of regression detector.")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    parser.add_argument("--output", default=get_default_output_json_path(BM_NAME), help="Specify the path of the output file")
    parser.add_argument("--debug", action="store_true", help="Save the debug output.")
    return parser.parse_known_args(args)

def run(args: List[str]):
    args, extra_args = parse_known_args(args)
    # If not specified, use the entire model set
    if not args.models:
        args.models = list_models()
    debug_output_dir = get_default_debug_output_dir(args.output) if args.debug else None
    devices = validate(parse_str_to_list(args.device), list_devices())
    tests = validate(parse_str_to_list(args.test), list_tests())
    batch_sizes = parse_str_to_list(args.bs)
    models = validate(parse_str_to_list(args.models), list_models())
    configs = generate_model_configs(devices, tests, batch_sizes, model_names=models, extra_args=extra_args)
    configs = init_output_dir(configs, debug_output_dir) if debug_output_dir else configs
    results = {}
    try:
        for config in configs:
            metrics = get_metrics(config)
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
    if args.device == 'cuda':
        import torch
        result["environ"]["device"] = torch.cuda.get_device_name()
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)
