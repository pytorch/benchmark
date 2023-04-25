"""
Test user-customized invoke function.
"""
import argparse
from typing import List
from ..utils import REPO_PATH, add_path, get_output_json, dump_output

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, load_model_isolated, TorchBenchModelConfig, \
                                                            list_devices, list_tests
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics

from typing import Optional

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", default="cuda", help="Devices to run, splited by comma.")
    parser.add_argument("--test", "-t", default="eval", help="Tests to run, splited by comma.")
    parser.add_argument("--model", "-m", default=None, type=str, help="Only run the specifice models, splited by comma.")
    return parser.parse_args(args)

def get_metrics(_config: TorchBenchModelConfig) -> List[str]:
    return ["latencies", "cpu_peak_mem", "gpu_peak_mem"]

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

def run(args: List[str]):
    args = parse_args(args)
    config = TorchBenchModelConfig(
        name=args.model,
        device=args.device,
        test=args.test,
        batch_size=None,
        jit=False,
        extra_args=[],
        extra_env=None,
    )
    result = run_config(config)
    print(result)
