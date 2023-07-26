"""
Run PyTorch nightly benchmarking.
"""
import re
import argparse
import itertools
import json
import math
import os
import yaml
import numpy

from typing import List, Tuple, Dict, Optional, Any
from ..utils import REPO_PATH, add_path, get_output_json, get_default_output_json_path
from . import BM_NAME

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, load_model_isolated, TorchBenchModelConfig, \
                                                            list_devices, list_tests
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_DELTA_THRESHOLD = 0.07
DEFAULT_TARGET_SCORE = 1000.0


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
        extra_args=[],
        extra_env=None,
    ) for device, test, model_name in cfgs]
    return result

def get_metrics(_config: TorchBenchModelConfig) -> List[str]:
    return ["latencies",]

def compute_score(results, reference_latencies: Dict[str, float]) -> float:
    # sanity checks
    latency_results = {k: v for k, v in results.items() if k.endswith("_latency")}
    test_set = set(latency_results.keys())
    reference_set = set(reference_latencies.keys())
    test_only_set = test_set.difference(reference_set)
    assert not test_only_set, f"Tests {test_only_set} only appears in result json, not in reference yaml."
    reference_only_set = reference_set.difference(test_set)
    assert not reference_only_set, f"Tests {reference_only_set} only appears in reference yaml, not in result json."
    # check that for every test in reference_latencies, we can find the corresponding tests in latency_results
    total_score = 0.0
    weight = 1.0 / len(reference_latencies)
    for key, ref_latency in reference_latencies.items():
        test_latency = latency_results[key]
        ref_latency = float(ref_latency)
        delta = (test_latency - ref_latency) / test_latency
        # If less than threshold, treat it as noise
        if abs(delta) <= DEFAULT_DELTA_THRESHOLD:
            test_latency = ref_latency
        total_score += weight * math.log(ref_latency / test_latency)
    score = math.exp(total_score) * DEFAULT_TARGET_SCORE
    return score

def result_to_output_metrics(results: List[Tuple[TorchBenchModelConfig, TorchBenchModelMetrics]]) -> Dict[str, float]:
    # metrics name examples:
    # test_eval[timm_regnet-cuda-eager]_latency
    # test_eval[timm_regnet-cuda-eager]_cmem
    # test_eval[timm_regnet-cuda-eager]_gmem
    result_metrics = {}
    for _config_id, (config, metrics) in enumerate(results):
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

def validate(candidates: List[str], choices: List[str]) -> List[str]:
    """Validate the candidates provided by the user is valid"""
    for candidate in candidates:
        assert candidate in choices, f"Specified {candidate}, but not in available list: {choices}."
    return candidates

def generate_model_configs_from_yaml(yaml_file: str) -> Tuple[List[TorchBenchModelConfig], Dict[str, float], Any]:
    yaml_file_path = os.path.join(CURRENT_DIR, yaml_file)
    with open(yaml_file_path, "r") as yf:
        config_obj = yaml.safe_load(yf)
    devices = config_obj["metadata"]["devices"]
    configs = []
    reference_latencies = {}
    for device in devices:
        for c in config_obj[device]:
            if not c["stable"]:
                continue
            config = TorchBenchModelConfig(
                name=c["model"],
                device=device,
                test=c["test"],
                batch_size=c["batch_size"] if "batch_size" in c else None,
                extra_args=[],
                extra_env=None,
            )
            configs.append(config)
            metrics_base = f"test_{config.test}[{config.name}-{config.device}-eager]"
            latency_metric_key = f"{metrics_base}_latency"
            reference_latencies[latency_metric_key] = c["median_latency"]
    return configs, reference_latencies, config_obj


def parse_test_name(test_name: str) -> TorchBenchModelConfig:
    regex = "test_(.*)\[(.*)-(.*)-eager\]"
    test, model, device = re.match(regex, test_name).groups()
    return TorchBenchModelConfig(
        name=model,
        device=device,
        test=test,
        batch_size=None,
        extra_args=[],
        extra_env=None,
    )

def generate_model_configs_from_bisect_yaml(bisect_yaml_file: str) -> List[TorchBenchModelConfig]:
    def _remove_suffix(test_name: str):
        index_last_underscore = test_name.rfind("_")
        return test_name[:index_last_underscore]
    with open(bisect_yaml_file, "r") as yf:
        bisect_obj = yaml.safe_load(yf)
    # remove the suffix
    bisect_tests = [ _remove_suffix(test_name) for test_name in bisect_obj["details"] ]
    bisect_tests = set(bisect_tests)
    configs = [ parse_test_name(test_name_str) for test_name_str in sorted(bisect_tests) ]
    return configs

def parse_str_to_list(candidates):
    if isinstance(candidates, list):
        return candidates
    candidates = list(map(lambda x: x.strip(), candidates.split(",")))
    return candidates

def run_config(config: TorchBenchModelConfig, dryrun: bool=False) -> Optional[TorchBenchModelMetrics]:
    """This function only handles NotImplementedError, all other errors will fail."""
    metrics = get_metrics(config)
    print(f"Running {config} ...", end='', flush=True)
    if dryrun:
        print(" [Skip: Dryrun]", flush=True)
        return None
    # We do not allow RuntimeError in this test
    try:
        # load the model instance in subprocess
        model = load_model_isolated(config)
        # get the model test metrics
        result: TorchBenchModelMetrics = get_model_test_metrics(model, metrics=metrics)
    except NotImplementedError as e:
        print(" [NotImplemented]", flush=True)
        return None
    print(" [Done]", flush=True)
    return result

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", default="cuda", help="Devices to run, splited by comma.")
    parser.add_argument("--test", "-t", default="eval", help="Tests to run, splited by comma.")
    parser.add_argument("--model", "-m", default=None, type=str, help="Only run the specifice models, splited by comma.")
    parser.add_argument("--config", "-c", default=None, help="YAML config to specify tests to run.")
    parser.add_argument("--run-bisect", help="Run with the output of regression detector.")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    parser.add_argument("--score", default=None, help="Generate score from the past run json only.")
    parser.add_argument("--output", default=get_default_output_json_path(BM_NAME), help="Specify the path of the output file")
    return parser.parse_args(args)

def run(args: List[str]):
    args = parse_args(args)
    if args.score:
        assert args.config, f"To compute score, you must specify the config YAML using --config."
        configs, reference_latencies, config_obj = generate_model_configs_from_yaml(args.config)
        with open(args.score, "r") as sp:
            run_result = json.load(sp)
        input_metrics = run_result["metrics"]
        score = compute_score(input_metrics, reference_latencies)
        score_version = config_obj["metadata"]["score_version"]
        score_name = f"{score_version}_score"
        print(f"TorchBench {score_name}: {score}.")
        exit(0)
    elif args.config:
        configs, reference_latencies, config_obj = generate_model_configs_from_yaml(args.config)
    elif args.run_bisect:
        configs = generate_model_configs_from_bisect_yaml(args.run_bisect)
        reference_latencies = None
    else:
        # If not specified, use the entire model set
        if not args.model:
            args.model = list_models()
        devices = validate(parse_str_to_list(args.device), list_devices())
        tests = validate(parse_str_to_list(args.test), list_tests())
        models = validate(parse_str_to_list(args.model), list_models())
        configs = generate_model_configs(devices, tests, model_names=models)
        reference_latencies = None
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
        if reference_latencies:
            score = compute_score(metrics, reference_latencies)
            score_version = config_obj["metadata"]["score_version"]
            score_name = f"{score_version}_score"
            metrics[score_name] = score
        result = get_output_json(BM_NAME, metrics)
        import torch
        result["environ"]["device"] = torch.cuda.get_device_name()
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=4)
