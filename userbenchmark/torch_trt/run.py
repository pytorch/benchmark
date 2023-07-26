import argparse
import traceback
import torch

import numpy as np

import json
import os
import time
from datetime import datetime
from typing import List

from torchbenchmark import (
    load_canary_model_by_name,
    load_model_by_name,
    list_models,
    ModelNotFoundError,
)


def cli(args: List[str]):
    """Parse input arguments, extracting model specification and batch size"""
    arg_parser = argparse.ArgumentParser(args)
    arg_parser.add_argument(
        "--model",
        help="Full or partial name of a model to run. If partial, picks the first match.",
        default="",
        type=str,
    )
    arg_parser.add_argument(
        "--bs",
        help="Input batch size to test.",
        default=1,
        type=int,
    )
    arg_parser.add_argument(
        "--num_warmup",
        help="Number of inference warmup iterations.",
        default=10,
        type=int,
    )
    arg_parser.add_argument(
        "--num_iter",
        help="Number of inference iterations for benchmarking.",
        default=100,
        type=int,
    )
    parsed_args, unknown = arg_parser.parse_known_args()

    return vars(parsed_args), unknown


def save_metrics(metrics):
    """Save metrics to a JSON file with formatted filename"""
    metrics_json = {
        "name": "torch_trt",
        "environ": {
            "metrics_version": "v0.1",
            "pytorch_git_version": torch.version.git_version,
        },
        "metrics": metrics,
    }

    # Obtain target save directory for JSON metrics from current save directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.normpath(
        os.path.join(current_dir, "../../.userbenchmark/torch_trt/")
    )

    os.makedirs(target_dir, exist_ok=True)

    # Format filename and path to save metrics
    metrics_file = "metrics-{}.json".format(
        datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    )
    metrics_save_path = os.path.join(target_dir, metrics_file)

    with open(metrics_save_path, "w") as f:
        json.dump(metrics_json, f, indent=4)


def run_single_model(
    Model,
    batch_size: int,
    extra_args: List[str],
    selected_ir: str,
    num_warmup: int,
    num_iter: int,
):
    """Run inference benchmarking on a single model"""
    # Build TorchBench model instance, with backend having the userbenchmark name
    # This invokes the torch_trt backend functionality directly
    model = Model(
        device="cuda",
        test="eval",
        batch_size=batch_size,
        extra_args=[
            "--backend",
        ]
        + extra_args,
    )

    metrics = run_one_step(model.invoke, model, num_warmup, num_iter, selected_ir)

    # Print dynamo compilation metrics, if there are any.
    try:
        if model.pt2_compilation_time:
            metrics[
                f"{model.name}.bs_{model.batch_size}.precision_{model.dargs.precision}."
                + f"ir_{selected_ir}.pt2_compilation_time"
            ] = model.pt2_compilation_time
    except:
        pass

    return metrics


def run_one_step(func, model, num_warmup, num_iter, selected_ir):
    # Warmup model inference
    for _ in range(num_warmup):
        func()

    result_summary = []

    # Run inference for the specified number of iterations
    for _ in range(num_iter):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Collect time_ns() instead of time() which does not provide better precision than 1
        # second according to https://docs.python.org/3/library/time.html#time.time.
        t0 = time.time_ns()
        start_event.record()
        func()
        end_event.record()
        torch.cuda.synchronize()
        t1 = time.time_ns()
        result_summary.append(
            (start_event.elapsed_time(end_event), (t1 - t0) / 1_000_000)
        )

    # Get median times for GPU and CPU Walltime
    gpu_time = np.median(list(map(lambda x: x[0], result_summary)))
    cpu_walltime = np.median(list(map(lambda x: x[1], result_summary)))

    if hasattr(model, "NUM_BATCHES"):
        median_gpu_time_per_batch = gpu_time / model.NUM_BATCHES
        median_cpu_walltime_per_batch = cpu_walltime / model.NUM_BATCHES
    else:
        median_gpu_time_per_batch = gpu_time
        median_cpu_walltime_per_batch = cpu_walltime

    metrics = {
        f"{model.name}.bs_{model.batch_size}.precision_{model.dargs.precision}."
        + f"ir_{selected_ir}.median_gpu_time_per_batch": median_gpu_time_per_batch,
        f"{model.name}.bs_{model.batch_size}.precision_{model.dargs.precision}."
        + f"ir_{selected_ir}.median_cpu_walltime_per_batch": median_cpu_walltime_per_batch,
    }

    return metrics


def run(args: List[str]):
    """Run inference and extract requested metrics"""
    parsed_args, unknown_args = cli(args)

    # Attempt to extract specified IR for logging purposes
    try:
        ir_idx = unknown_args.index("--ir")
        selected_ir = unknown_args[ir_idx + 1]
    except (ValueError, IndexError):
        selected_ir = "default"

    # Parse model string if specified, otherwise run all models
    # Adapted from benchmark/run.py
    if parsed_args["model"]:
        try:
            Model = load_model_by_name(parsed_args["model"])
        except ModuleNotFoundError:
            traceback.print_exc()
            exit(-1)
        except ModelNotFoundError:
            print(
                f"Warning: The model {parsed_args['model']} cannot be found at core set."
            )
        if not Model:
            try:
                Model = load_canary_model_by_name(parsed_args["model"])
            except ModuleNotFoundError:
                traceback.print_exc()
                exit(-1)
            except ModelNotFoundError:
                print(
                    f"Error: The model {parsed_args['model']} cannot be found at either core or canary model set."
                )
                exit(-1)

        all_metrics = run_single_model(
            Model,
            parsed_args["bs"],
            unknown_args,
            selected_ir,
            parsed_args["num_warmup"],
            parsed_args["num_iter"],
        )

    else:
        all_metrics = {}

        for Model in list_models():
            metrics = run_single_model(
                Model,
                parsed_args["bs"],
                unknown_args,
                selected_ir,
                parsed_args["num_warmup"],
                parsed_args["num_iter"],
            )
            all_metrics = {**all_metrics, **metrics}

    save_metrics(all_metrics)
