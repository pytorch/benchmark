"""
Generate model traces that are runnable in Tritonbench.
Example trace:
"""

import argparse
import itertools
import os
from typing import Dict, List

from torchbenchmark.util.experiment.instantiator import (
    list_extended_models,
    list_models,
    load_model,
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
    parser.add_argument(
        "--suites",
        type=str,
        default="all",
        help="Suites to trace, split by comma. Default is all.",
    )
    parser.add_argument(
        "--skip", type=str, help="models to skip, split by comma. Default is none."
    )
    parser.add_argument(
        "--bypass-existing", action="store_true", help="Bypass existing trace files."
    )
    parser.add_argument(
        "--non-isolated", action="store_true", help="Run in non-isolated mode."
    )
    parser.add_argument("--output", type=str, help="Output directory.", required=True)
    return parser


# Function hat runs in the worker process.
# Must include all imports
# Can only return None, return values saved in the output file
def test_run_model(model) -> None:
    import os

    from torchbenchmark.util.dispatch import OperatorInputsMode

    output_dir = os.environ.get("TORCHBENCH_MODEL_TRACE_OUTPUT_DIR", "")
    if output_dir and not output_dir.endswith("/"):
        output_dir += "/"
    output_filename = f"{output_dir}{model.name}_{model.test}.json"
    with OperatorInputsMode(output_filename=output_filename):
        model.invoke()
    return


def trace_model(
    cfg: TorchBenchModelConfig, isolated: bool = True, bypass_existing=False
):
    output_dir = cfg.extra_env["TORCHBENCH_MODEL_TRACE_OUTPUT_DIR"]
    if output_dir and not output_dir.endswith("/"):
        output_dir += "/"
    output_filename = f"{output_dir}{cfg.name}_{cfg.test}.json"
    if (
        os.path.exists(output_filename)
        and os.path.getsize(output_filename)
        and bypass_existing
    ):
        return
    if isolated:
        model_task = load_model_isolated(cfg)
        model_task.run(test_run_model)
    else:
        model = load_model(cfg)
        test_run_model(model)


def run_models(models: List[str], args: argparse.Namespace, extra_env: Dict[str, str]):
    modes = args.mode.split(",")
    model_args = list(itertools.product(models, modes))
    cfgs = [
        generate_model_config(model_name, mode, extra_env)
        for model_name, mode in model_args
    ]
    for cfg in cfgs:
        print(f"tracing {cfg.name}-{cfg.test} ...", end="")
        try:
            trace_model(
                cfg,
                isolated=not args.non_isolated,
                bypass_existing=args.bypass_existing,
            )
            print("[done]")
        except NotImplementedError as e:
            print(f"[not_implemented] {e}")


def run_model_suite(args: argparse.Namespace, suite: str, extra_env: Dict[str, str]):
    suite_name = "hf" if suite == "huggingface" else suite
    extra_env["TORCHBENCH_MODEL_TRACE_OUTPUT_DIR"] = (
        f"{extra_env['TORCHBENCH_MODEL_TRACE_OUTPUT_DIR']}/{suite_name}_train/"
    )
    if suite == "torchbench":
        models = list_models()
    else:
        models = list_extended_models(suite_name=suite)
    if args.skip:
        models = [model for model in models if model not in args.skip.split(",")]
    run_models(models, args, extra_env)


def run(args: List[str]):
    parser = get_parser()
    args: argparse.Namespace = parser.parse_args(args)
    suites = args.suites.split(",")
    if suites == ["all"]:
        suites = ["torchbench", "huggingface", "timm"]
    extra_env = {"TORCHBENCH_MODEL_TRACE_OUTPUT_DIR": args.output}
    if args.models:
        models = args.models.split(",")
        run_models(models, args, extra_env)
    else:
        for suite in suites:
            run_model_suite(args, suite, extra_env)
