"""
Generate model traces that are runnable in Tritonbench.
Example trace:
"""

import argparse
import itertools
from typing import List

from torchbenchmark.util.experiment.instantiator import (
    list_models,
    load_model_isolated,
    TorchBenchModelConfig,
)


def generate_model_config(model_name: str, mode: str) -> TorchBenchModelConfig:
    return TorchBenchModelConfig(
        name=model_name,
        device="cuda",
        test=mode,
        batch_size=None,
        extra_args=[],
        extra_env=None,
    )


class ModelTrace:
    pass


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", type=str, default=None, help="Model to trace, split by comma."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Mode to trace, default is train.",
    )
    parser.add_argument("--output", type=str, help="Output directory.")
    return parser


def trace_model(cfg: TorchBenchModelConfig) -> ModelTrace:
    task = load_model_isolated(cfg)
    return ModelTrace()


def run(args: List[str]):
    parser = get_parser()
    args: argparse.Namespace = parser.parse_args(args)
    models = args.models.split(",") if args.models else list_models()
    modes = args.mode.split(",") if args.mode else ["train"]
    model_args = list(itertools.product(models, modes))
    cfgs = [generate_model_config(model_name, mode) for model_name, mode in model_args]
    traces = [trace_model(cfg) for cfg in cfgs]
