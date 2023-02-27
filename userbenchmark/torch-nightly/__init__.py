"""
Run PyTorch nightly benchmarking.
"""
import argparse
import itertools

from typing import List
from ..utils import REPO_PATH, add_path, get_output_dir, get_output_json, dump_output

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, load_model, TorchBenchModelConfig
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics

BM_NAME = "torch-nightly"

def generate_ootb_model_configs(devices: List[str], tests: List[str], model_names: List[str]) -> List[TorchBenchModelConfig]:
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

def parse_str_to_list(tests: str):
    pass

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="cuda", choices=["cuda", "cpu"], help="The name of the config to run.")
    parser.add_argument("--tests", choices=["train", "eval", "train,eval", "eval,train"], help="Tests to run.")
    parser.add_argument("--models", "-m", default=None, nargs="+", help="Only run the specifice models, model names splited with space.")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    return parser.parse_args(args)

def run(args: List[str]):
    args = parse_args(args)
    benchmark_configs = generate_ootb_model_configs(parse_str_to_list(args.devices), parse_str_to_list(args.tests), parse_str_to_list(args.models))
    for benchmark_config in benchmark_configs:
        pass