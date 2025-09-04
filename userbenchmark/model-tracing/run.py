"""
Generate model traces that are runnable in Tritonbench.
Example trace:
"""

import argparse
import itertools
from typing import Dict, List

from torchbenchmark.util.experiment.instantiator import (
    list_extended_models,
    list_models,
    load_model_isolated,
    TorchBenchModelConfig,
)


def generate_model_config(
    model_name: str, mode: str, extra_env: Dict[str, str]
) -> TorchBenchModelConfig:
    return TorchBenchModelConfig(
        name=model_name,
        device="cuda",
        test=mode,
        batch_size=None,
        extra_args=[],
        extra_env=extra_env,
    )


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


# Function hat runs in the worker process.
# Must include all imports
# Can only return None, return values saved in the output file
def test_run_model(model) -> None:
    import os

    from torchbenchmark.util.dispatch import OperatorInputsMode

    output_dir = os.environ.get("TORCHBENCH_MODEL_TRACE_OUTPUT_DIR", "")
    output_filename = f"{output_dir}{model.name}_{model.test}.txt"
    with OperatorInputsMode(output_filename=output_filename):
        model.invoke()
    return


def trace_model(cfg: TorchBenchModelConfig):
    model_task = load_model_isolated(cfg)
    model_task.run(test_run_model)


def run_models(models: List[str], args: argparse.Namespace, extra_env: Dict[str, str]):
    modes = args.mode.split(",")
    model_args = list(itertools.product(models, modes))
    cfgs = [
        generate_model_config(model_name, mode, extra_env)
        for model_name, mode in model_args
    ]
    traces = [trace_model(cfg) for cfg in cfgs]


def run_model_suite(args: argparse.Namespace, suite: str):
    extra_env = {}
    extra_env["TORCHBENCH_MODEL_TRACE_OUTPUT_DIR"] = (
        f"{extra_env['TORCHBENCH_MODEL_TRACE_OUTPUT_DIR']}/{suite}_train"
    )
    if suite == "torchbench":
        models = list_models()
    else:
        models = list_extended_models(suite_name=suite)
    run_models(models, args, extra_env)


def run(args: List[str]):
    parser = get_parser()
    args: argparse.Namespace = parser.parse_args(args)
    extra_env = (
        {"TORCHBENCH_MODEL_TRACE_OUTPUT_DIR": args.output} if args.output else {}
    )
    if args.models:
        models = args.models.split(",")
        run_models(models, args, extra_env)
    else:
        # run for all models
        run_model_suite(args, "torchbench")
        run_model_suite(args, "huggingface")
        run_model_suite(args, "timm")
