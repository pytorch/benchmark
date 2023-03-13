"""
Run PyTorch nightly benchmarking.
"""
import argparse
import itertools
import yaml

from typing import List, Tuple, Dict
from ..utils import REPO_PATH, add_path, get_output_dir, get_output_json, dump_output

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, load_model, TorchBenchModelConfig, \
                                                            list_devices, list_tests
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics

BM_NAME = "torch-nightly"

def generate_model_configs(devices: List[str], tests: List[str], model_names: List[str]) -> List[TorchBenchModelConfig]:
    """Use the default batch size and default mode."""
    if not model_names:
        model_names = list_models()
    cfgs = itertools.product(*[devices, tests, model_names])
    result = [TorchBenchModelConfig(
        name=model_name,
        device=device,
        test=test,
        batch_size=None,
        jit=False,
        extra_args=[],
        extra_env=None,
    ) for device, test, model_name in cfgs]
    return result

def get_metrics(config: TorchBenchModelConfig) -> List[str]:
    if config.device == "cpu":
        return ["latencies", "cpu_peak_mem"]
    elif config.device == "cuda":
        return ["latencies", "cpu_peak_mem", "gpu_peak_mem"]
    else:
        return ["latencies"]

def result_to_output_metrics(results: List[Tuple[TorchBenchModelConfig, TorchBenchModelMetrics]]) -> Dict[str, float]:
    pass

def dump_result_to_json(metrics):
    result = get_output_json(BM_NAME, metrics)
    dump_output(BM_NAME, result)

def validate(candidates: List[str], choices: List[str]) -> List[str]:
    """Validate the candidates provided by the user is valid"""
    for candidate in candidates:
        assert candidate in choices, f"Specified {candidate}, but not in available list: {choices}."
    return candidates

def filter_yaml_to_configs(filter_config_file: str) -> List[TorchBenchModelConfig]:
    filter_obj = yaml.safe_load(filter_config_file)
    devices = filter_obj.keys()
    configs = []
    for device in devices:
        c = filter_obj[device]
        config = TorchBenchModelConfig(
            name=c["model"],
            device=device,
            test=c["test"],
            batch_size=c["batch_size"] if "batch_size" in c else None,
            jit=c["jit"] if "jit" in c else False,
            extra_args=[],
            extra_env=None,
        )
        configs.append(config)
    return configs

def parse_str_to_list(candidates: str):
    candidates = list(map(lambda x: x.strip(), candidates.split(",")))
    return candidates

def run_config(config: TorchBenchModelConfig, dryrun: bool=False) -> TorchBenchModelMetrics:
    metrics = get_metrics(config)

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", default="cuda", help="Devices to run, splited by comma.")
    parser.add_argument("--test", "-t", default="eval", help="Tests to run, splited by comma.")
    parser.add_argument("--model", "-m", default=None, nargs="+", help="Only run the specifice models, splited by comma.")
    parser.add_argument("--filter", default=None, help="YAML config to filter unstable tests.")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    return parser.parse_args(args)

def run(args: List[str]):
    args = parse_args(args)
    # If not specified, use the entire model set
    if not args.model:
        args.model = list_models()
    devices = validate(parse_str_to_list(args.device), list_devices())
    tests = validate(parse_str_to_list(args.test), list_tests())
    models = validate(parse_str_to_list(args.model), list_models())
    configs = generate_model_configs(devices, tests, model_names=models)
    if args.filter:
        filters = filter_yaml_to_configs(args.filter)
    configs = list(filter(lambda x: not x in filters, configs))
    results = []
    for config in configs:
        metrics = run_config(config, dryrun=args.dryrun)
        results.append([config, metrics])
    metrics = result_to_output_metrics(results)
    dump_result_to_json(metrics, args.output)
