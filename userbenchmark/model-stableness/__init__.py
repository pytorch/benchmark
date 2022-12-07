import itertools
from typing import List
import argparse

from ..utils import REPO_PATH, add_path

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, TorchBenchModelConfig

def generate_model_config(model_name: str) -> List[TorchBenchModelConfig]:
    devices = ["cpu", "cuda"]
    tests = ["train", "eval"]
    cfgs = itertools.product(*[devices, tests])
    result = [TorchBenchModelConfig(
        name=model_name,
        device=device,
        test=test,
        batch_size=None,
        jit=False,
        extra_args=[],
        extra_env=None,
    ) for device, test in cfgs]
    return result

def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu", help="Specify the device.")
    parser.add_argument("-t", "--test", default="eval", help="Specify the test.")
    parser.add_argument("-o", "--output", type=str, help="The default output json file.")
    args, unknown_args = parser.parse_known_args(args)
    return args, unknown_args

def generate_filter(args: argparse.Namespace):
    pass

def run(args: List[str]):
    args, unknown_args = parse_args(args)
    models = list_models()
    cfgs = list(itertools.chain(*map(generate_model_config, models)))
    for cfg in cfgs:
        print(cfg)
