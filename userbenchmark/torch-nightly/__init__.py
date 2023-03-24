"""
Run PyTorch nightly benchmarking.
"""
import argparse
import itertools
import os
import yaml
import numpy

from typing import List, Tuple, Dict, Optional
from ..utils import REPO_PATH, add_path, get_output_json, dump_output

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, load_model_isolated, TorchBenchModelConfig, \
                                                            list_devices, list_tests
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics

BM_NAME = "torch-nightly"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

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

def get_metrics(_config: TorchBenchModelConfig) -> List[str]:
    return ["latencies", "cpu_peak_mem", "gpu_peak_mem"]

def result_to_output_metrics(results: List[Tuple[TorchBenchModelConfig, TorchBenchModelMetrics]]) -> Dict[str, float]:
    # metrics name examples:
    # test_eval[timm_regnet-cuda-eager]_latency
    # test_eval[timm_regnet-cuda-eager]_cmem
    # test_eval[timm_regnet-cuda-eager]_gmem
    result_metrics = {}
    for config, metrics in results:
        metrics_base = f"test_{config.test}[{config.name}-{config.device}-eager]"
        latency_metric = f"{metrics_base}_latency"
        median_latency = numpy.median(metrics.latencies)
        assert median_latency, f"Run failed for metric {latency_metric}"
        result_metrics[latency_metric] = median_latency
        if metrics.cpu_peak_mem:
            cpu_peak_mem = f"{metrics_base}_cmem"
            result_metrics[cpu_peak_mem] = metrics.cpu_peak_mem
        if metrics.gpu_peak_mem:
            gpu_peak_mem = f"{metrics_base}_gmem"
            result_metrics[gpu_peak_mem] = metrics.gpu_peak_mem
    return result_metrics

def dump_result_to_json(metrics):
    result = get_output_json(BM_NAME, metrics)
    dump_output(BM_NAME, result)

def validate(candidates: List[str], choices: List[str]) -> List[str]:
    """Validate the candidates provided by the user is valid"""
    for candidate in candidates:
        assert candidate in choices, f"Specified {candidate}, but not in available list: {choices}."
    return candidates

def generate_model_configs_from_yaml(yaml_file: str) -> List[TorchBenchModelConfig]:
    yaml_file_path = os.path.join(CURRENT_DIR, yaml_file)
    with open(yaml_file_path, "r") as yf:
        config_obj = yaml.safe_load(yf)
    devices = config_obj.keys()
    configs = []
    for device in devices:
        for c in config_obj[device]:
            if not c["stable"]:
                continue
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

def parse_str_to_list(candidates):
    if isinstance(candidates, list):
        return candidates
    candidates = list(map(lambda x: x.strip(), candidates.split(",")))
    return candidates

def run_config(config: TorchBenchModelConfig, dryrun: bool=False) -> Optional[TorchBenchModelMetrics]:
    """This function only handles NotImplementedError, all other errors will fail."""
    metrics = get_metrics(config)
    print(f"Running {config} ...", end='')
    if dryrun:
        return None
    # We do not allow RuntimeError in this test
    try:
        # load the model instance within the same process
        model = load_model_isolated(config)
        # get the model test metrics
        result: TorchBenchModelMetrics = get_model_test_metrics(model, metrics=metrics)
    except NotImplementedError as e:
        print(" [NotImplemented]")
        return None
    print(" [Done]")
    return result

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", default="cuda", help="Devices to run, splited by comma.")
    parser.add_argument("--test", "-t", default="eval", help="Tests to run, splited by comma.")
    parser.add_argument("--model", "-m", default=None, type=str, help="Only run the specifice models, splited by comma.")
    parser.add_argument("--config", "-c", default=None, help="YAML config to specify tests to run.")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    return parser.parse_args(args)

def run(args: List[str]):
    args = parse_args(args)
    if args.config:
        configs = generate_model_configs_from_yaml(args.config)
    else:
        # If not specified, use the entire model set
        if not args.model:
            args.model = list_models()
        devices = validate(parse_str_to_list(args.device), list_devices())
        tests = validate(parse_str_to_list(args.test), list_tests())
        models = validate(parse_str_to_list(args.model), list_models())
        configs = generate_model_configs(devices, tests, model_names=models)
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
        dump_result_to_json(metrics)
