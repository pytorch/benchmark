"""
Run PyTorch nightly benchmarking.
"""
import argparse
import itertools
import yaml
import numpy

from typing import List, Tuple, Dict, Optional
from ..utils import REPO_PATH, add_path, get_output_json, dump_output

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, load_model_isolated, TorchBenchModelConfig, \
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
    # metrics name examples:
    # test_eval[timm_regnet-cuda-eager]_latency
    # test_eval[timm_regnet-cuda-eager]_cmem
    # test_eval[timm_regnet-cuda-eager]_gmem
    metrics = {}
    for config, metrics in results:
        metrics_base = f"test_{config.test}[{config.name}-{config.device}-eager]"
        latency_metric = f"{metrics_base}_latency"
        median_latency = numpy.median(metrics.latencies)
        if median_latency:
            metrics[latency_metric] = median_latency
        else:
            # The run has failed
            metrics[latency_metric] = -1.0
        if metrics.cpu_peak_mem:
            cpu_peak_mem = f"{metrics_base}_cmem"
            metrics[cpu_peak_mem] = metrics.cpu_peak_mem
        if metrics.gpu_peak_mem:
            gpu_peak_mem = f"{metrics_base}_gmem"
            metrics[gpu_peak_mem] = metrics.gpu_peak_mem
    return metrics

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

def run_config(config: TorchBenchModelConfig, dryrun: bool=False) -> Optional[TorchBenchModelMetrics]:
    """This function only handles NotImplementedError, all other errors will fail."""
    metrics = get_metrics(config)
    try:
        # load the model instance within the same process
        model = load_model_isolated(config)
        # get the model test metrics
        result: TorchBenchModelMetrics = get_model_test_metrics(model, metrics=metrics)
    except NotImplementedError as e:
        return None
    return result

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
    try:
        for config in configs:
            metrics = run_config(config, dryrun=args.dryrun)
            if metrics:
                results.append([config, metrics])
    except KeyboardInterrupt:
        print("User keyboard interrupted!")
    if not args.dryrun:
        metrics = result_to_output_metrics(results)
        dump_result_to_json(metrics, args.output)
