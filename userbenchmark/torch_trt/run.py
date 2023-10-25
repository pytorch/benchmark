import argparse
import json
import os
import time
import traceback
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from torchbenchmark import (
    ModelNotFoundError,
    ModelTask,
    load_canary_model_by_name,
    load_model_by_name,
)
from torchbenchmark.util.experiment.instantiator import (
    TorchBenchModelConfig,
    list_models,
    load_model_isolated,
)
from torchbenchmark.util.model import BenchmarkModel


def cli(args: List[str]):
    """Parse input arguments, extracting model specification and batch size"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model",
        help="Full or partial name of a model to run. If partial, picks the first match.",
        default="",
        type=str,
    )
    arg_parser.add_argument(
        "--bs",
        help="Input batch size to test.",
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


def save_metrics(metrics, error_logs):
    """Save metrics to a JSON file with formatted filename.
    
    Save error logs to a text file
    """
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
    curr_time = time.time()
    metrics_file = "metrics-{}.json".format(
        datetime.fromtimestamp(curr_time).strftime("%Y%m%d%H%M%S")
    )
    error_logs_file = "error-logs-{}.txt".format(
        datetime.fromtimestamp(curr_time).strftime("%Y%m%d%H%M%S")
    )

    metrics_save_path = os.path.join(target_dir, metrics_file)
    error_save_path = os.path.join(target_dir, error_logs_file)

    with open(metrics_save_path, "w") as f:
        json.dump(metrics_json, f, indent=4)

    with open(error_save_path, "w") as f:
        for model, log in error_logs.items():
            f.write(f"Model: {model} failed with error\n\n")
            f.write(log)
            f.write("\n++++++++++\n")


def run_single_model(
    model: Union[BenchmarkModel, ModelTask],
    selected_ir: str,
    num_warmup: int,
    num_iter: int,
):
    """Run inference benchmarking on a single model"""

    # Get basic metrics for the model
    metrics = run_one_step(model.invoke, model, num_warmup, num_iter, selected_ir)

    # Get PT2 compilation time for the model
    try:
        if isinstance(model, ModelTask):
            pt2_compilation_time = model.get_model_attribute("pt2_compilation_time")
            name = model.get_model_attribute("name")
            batch_size = model.get_model_attribute("batch_size")
            precision = model.get_model_attribute("dargs", "precision")
        else:
            pt2_compilation_time = getattr(model, "pt2_compilation_time", None)
            pt2_graph_breaks = getattr(model, "pt2_graph_breaks", None)
            name = getattr(model, "name", None)
            batch_size = getattr(model, "batch_size", None)
            precision = getattr(model, "precision", None)

        if pt2_compilation_time is not None and pt2_compilation_time:
            metrics[
                f"{name}.bs_{batch_size}.precision_{precision}."
                + f"ir_{selected_ir}.pt2_compilation_time"
            ] = pt2_compilation_time

        if pt2_graph_breaks is not None and pt2_graph_breaks:
            metrics[
                f"{name}.bs_{batch_size}.precision_{precision}."
                + f"ir_{selected_ir}.pt2_graph_breaks"
            ] = pt2_graph_breaks
    except:
        pass

    return metrics


def run_one_step(
    func,
    model: Union[BenchmarkModel, ModelTask],
    num_warmup: int,
    num_iter: int,
    selected_ir: str,
):
    """Run one step of inference benchmarking on a single model"""
    # Warmup model inference
    for _ in range(num_warmup):
        func()

    result_summary = []

    # Run inference for the specified number of iterations
    for _ in range(num_iter):
        torch.cuda.synchronize()

        # Collect time_ns() instead of time() which does not provide better precision than 1
        # second according to https://docs.python.org/3/library/time.html#time.time.
        t0 = time.time_ns()
        func()
        torch.cuda.synchronize()
        t1 = time.time_ns()
        result_summary.append((t1 - t0) / 1_000_000)

    # Get median times for GPU and CPU Walltime
    latency = np.median(result_summary)

    # Differentiate model attribute access based on input type
    if isinstance(model, ModelTask):
        num_batches = model.get_model_attribute("NUM_BATCHES")
        name = model.get_model_attribute("name")
        batch_size = model.get_model_attribute("batch_size")
        precision = model.get_model_attribute("dargs", "precision")
    else:
        num_batches = getattr(model, "NUM_BATCHES", None)
        name = getattr(model, "name", None)
        batch_size = getattr(model, "batch_size", None)
        precision = getattr(model, "precision", None)

    if num_batches is not None:
        median_latency_per_batch = latency / num_batches
    else:
        median_latency_per_batch = -1.0

    # Store metrics as dictionary
    metrics = {
        f"{name}.bs_{batch_size}.precision_{precision}."
        + f"ir_{selected_ir}.median_latency_ms": latency,
        f"{name}.bs_{batch_size}.precision_{precision}."
        + f"ir_{selected_ir}.median_latency_ms_per_batch": median_latency_per_batch,
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
        # If no IR was specified, default to torch.compile
        selected_ir = "torch_compile"
        unknown_args.append("--ir")
        unknown_args.append(selected_ir)


    error_logs = {}

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

        # For single models, use a BenchmarkModel instance
        model = Model(
            device="cuda",
            test="eval",
            batch_size=parsed_args["bs"],
            extra_args=[
                "--backend",
            ]
            + unknown_args
            + ["--truncate_long_and_double"],
        )

        all_metrics = run_single_model(
            model,
            selected_ir,
            parsed_args["num_warmup"],
            parsed_args["num_iter"],
        )

    else:
        all_metrics = {}

        # For all models, use ModelTask instances
        for model_name in list_models():
            # Add optional argument `truncate_long_and_double`, as required for many models
            config = TorchBenchModelConfig(
                name=model_name,
                test="eval",
                device="cuda",
                batch_size=parsed_args["bs"],
                extra_args=[
                    "--backend",
                ]
                + unknown_args
                + ["--truncate_long_and_double"],
            )

            try:
                Model = load_model_isolated(config=config)
                print(f"\nLoading model {model_name} succeeded.\n")
                metrics = run_single_model(
                    Model,
                    selected_ir,
                    parsed_args["num_warmup"],
                    parsed_args["num_iter"],
                )

                # Delete model instance and clean up workspace
                del Model

            # Exceptions raised can be reported in a verbose manner
            except Exception as e:
                traceback.print_exc()
                print(
                    f"\nBenchmarking model {model_name} failed with:\n{e}\nSkipping the model.\n"
                )
                error_logs[model_name] = f"Failed to run benchmark: {traceback.format_exc()}"

            # Halt further model runs on KeyboardInterrupt
            except KeyboardInterrupt:
                break

            # Any recoverable subprocess errors are caught and reported
            except:
                print(
                    f"\nBenchmarking model {model_name} failed.\nSkipping the model.\n"
                )
                error_logs[model_name] = "Failed to run benchmark: Other Python error"

            all_metrics = {**all_metrics, **metrics}

    save_metrics(all_metrics, error_logs)
