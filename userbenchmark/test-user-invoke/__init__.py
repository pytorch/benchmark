"""
Test user-customized invoke function.
"""
import argparse
from typing import List
from ..utils import REPO_PATH, add_path, get_output_json, dump_output

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, load_model_isolated, TorchBenchModelConfig, \
                                                            list_devices, list_tests, inject_model_invoke
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics

from typing import Optional

def user_defined_invoke(self):
    print(f"Model {self.name} invoke has been replaced!")

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", default="cuda", help="Devices to run, splited by comma.")
    parser.add_argument("--test", "-t", default="eval", help="Tests to run, splited by comma.")
    parser.add_argument("--bs", type=int, default=1, help="Test batch size")
    parser.add_argument("--model", "-m", default=None, type=str, help="Only run the specifice models, splited by comma.")
    parser.add_argument("--inject", action="store_true", help="Inject user defined invoke function to the model.")
    return parser.parse_args(args)

def get_metrics(_config: TorchBenchModelConfig) -> List[str]:
    return ["latencies"]

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
        inject_model_invoke(model, user_defined_invoke)
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
        batch_size=args.bs,
        jit=False,
        extra_args=[],
        extra_env=None,
    )
    result = run_config(config)
    print(result)
