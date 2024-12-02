"""
Run TorchBench test benchmarking.
"""

import argparse
import ast
import itertools
import json
import os
import pathlib
import re
import shutil

from typing import Dict, List, Optional, Set, Union

import numpy
import yaml

from ..utils import (
    add_path,
    get_default_debug_output_dir,
    get_default_output_json_path,
    get_output_json,
    REPO_PATH,
)
from torchbenchmark import get_metadata_from_yaml, ModelTask, REPO_PATH

from . import BM_NAME

# Some of the models have very heavyweight setup, so we have to set a very
# generous limit. That said, we don't want the entire test suite to hang if
# a single test encounters an extreme failure, so we give up after a test is
# unresponsive to 5 minutes by default. (Note: this does not require that the
# entire test case completes in 5 minutes. It requires that if the worker is
# unresponsive for 5 minutes the parent will presume it dead / incapacitated.)
TIMEOUT = int(os.getenv("TIMEOUT", 300))  # Seconds

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import (
        list_devices,
        list_extended_models,
        list_models,
        list_tests,
        load_model_isolated,
        TorchBenchModelConfig,
    )
    from torchbenchmark.util.experiment.metrics import (
        get_model_accuracy,
        get_model_test_metrics,
        TorchBenchModelMetrics,
    )

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def config_to_str(config: TorchBenchModelConfig) -> str:
    metrics_base = (
        f"model={config.name}, test={config.test}, device={config.device},"
        + f" bs={config.batch_size}, extra_args={config.extra_args}"
    )
    return metrics_base


def str_to_config(metric_name: str) -> TorchBenchModelConfig:
    regex = "model=(.*), test=(.*), device=(.*), bs=(.*), extra_args=(.*), metric=(.*)"
    model, test, device, batch_size, extra_args, _metric = re.match(
        regex, metric_name
    ).groups()
    extra_args = ast.literal_eval(extra_args)
    batch_size = ast.literal_eval(batch_size)
    return TorchBenchModelConfig(
        name=model,
        test=test,
        device=device,
        batch_size=batch_size,
        extra_args=extra_args,
    )


def generate_model_configs(
    devices: List[str],
    tests: List[str],
    batch_sizes: List[str],
    model_names: List[str],
    skip_models: List[str],
    extra_args: List[str],
) -> List[TorchBenchModelConfig]:
    """Use the default batch size and default mode."""
    cfgs = itertools.product(*[devices, tests, batch_sizes, model_names])
    result = [
        TorchBenchModelConfig(
            name=model_name,
            device=device,
            test=test,
            batch_size=None if not batch_size else int(batch_size),
            extra_args=extra_args,
            extra_env=None,
            skip=model_name in skip_models
        )
        for device, test, batch_size, model_name in cfgs
    ]
    return result


def generate_model_configs_from_bisect_yaml(
    bisect_yaml: str,
) -> List[TorchBenchModelConfig]:
    with open(bisect_yaml, "r") as fp:
        bisect = yaml.safe_load(fp)
    result = list(map(lambda x: str_to_config(x), bisect["details"].keys()))
    return result


def init_output_dir(
    configs: List[TorchBenchModelConfig], output_dir: pathlib.Path
) -> List[TorchBenchModelConfig]:
    result = []
    for config in configs:
        config_str = config_to_str(config)
        config.output_dir = output_dir.joinpath(config_str)
        if config.output_dir.exists():
            shutil.rmtree(config.output_dir)
        config.output_dir.mkdir(parents=True)
        result.append(config)
    return result


def get_metrics(config: TorchBenchModelConfig) -> List[str]:
    if "--accuracy" in config.extra_args:
        return ["accuracy"]
    if "--memleak" in config.extra_args:
        return ["memleak"]
    return ["latencies", "cpu_peak_mem", "gpu_peak_mem"]


def validate(candidates: List[str], choices: Union[Set[str], List[str]]) -> List[str]:
    """Validate the candidates provided by the user is valid"""
    for candidate in candidates:
        assert (
            candidate in choices
        ), f"Specified {candidate}, but not in available list: {choices}."
    return candidates


def parse_str_to_list(candidates: Optional[str]) -> List[str]:
    if isinstance(candidates, list):
        return candidates
    elif candidates == None:
        return [""]
    candidates = list(map(lambda x: x.strip(), candidates.split(",")))
    return candidates


def metrics_to_dict(
    metrics: Union[TorchBenchModelMetrics, Dict[str, str]]
) -> Dict[str, Union[str, float]]:
    if isinstance(metrics, TorchBenchModelMetrics):
        pass
    return metrics


def run_config(
    config: TorchBenchModelConfig, metrics: List[str], dryrun: bool = False
) -> Dict[str, Union[str, float]]:
    """This function only handles NotImplementedError, all other errors will fail."""
    print(f"Running {config} ...", end="", flush=True)
    if dryrun:
        print(" [skip_by_dryrun]", flush=True)
        return dict.fromkeys(metrics, "skip_by_dryrun")
    if config.skip:
        print(" [skip]", flush=True)
        return dict.fromkeys(metrics, "skip")
    # We do not allow RuntimeError in this test
    try:
        # load the model instance in subprocess
        model = load_model_isolated(config)
        # get the model test metrics
        metrics_output: TorchBenchModelMetrics = get_model_test_metrics(
            model, metrics=metrics
        )
        result = {}
        for metric in metrics:
            if metric == "latencies" and metrics_output.latencies:
                result[metric] = numpy.median(metrics_output.latencies)
            else:
                result[metric] = getattr(metrics_output, metric, None)
                result[metric] = "failed" if result[metric] == None else result[metric]
        print(" [done]", flush=True)
        return result
    except NotImplementedError as e:
        print(" [not_implemented]", flush=True)
        return dict.fromkeys(metrics, "not_implemented")
    except OSError as e:
        print(" [oserror]", flush=True)
        return dict.fromkeys(metrics, str(e))
    except KeyboardInterrupt:
        print(" [user_interrupt]")
        exit(1)
    except Exception as e:
        print(" [runtime_error]", flush=True)
        return dict.fromkeys(metrics, str(e))


def run_config_memleak(
    config: TorchBenchModelConfig, dryrun: bool = False
) -> Dict[str, str]:
    def assertEqual(x, y):
        assert x == y, f"{x} != {y}"

    model_name = config.name
    model_path = os.path.join(REPO_PATH, "torchbenchmark", "models", model_name)
    metadata = get_metadata_from_yaml(model_path)
    task = ModelTask(model_name, timeout=TIMEOUT)
    allow_customize_batch_size = task.get_model_attribute(
        "ALLOW_CUSTOMIZE_BSIZE", classattr=True
    )
    # to speedup test, use batch size 1 if possible
    batch_size = 1 if allow_customize_batch_size else None
    if dryrun:
        print(" [skip_by_dryrun] ", flush=True)
        return {"memleak": "skip_by_dryrun"}
    if config.skip:
        print(" [skip]", flush=True)
        return {"memleak": "skip"}
    try:
        with task.watch_cuda_memory(
            skip=False,
            assert_equal=assertEqual,
        ):
            task.make_model_instance(
                test=config.test,
                device=config.device,
                batch_size=batch_size,
            )
            task.invoke()
            if config.test == "train":
                task.check_details_train(device=config.device, md=metadata)
            else:
                task.check_details_eval(device=config.device, md=metadata)
                task.check_eval_output()
            task.del_model_instance()
            result = {"memleak": "False"}
    except NotImplementedError as e:
        result = {"memleak": "not_implemented"}
    except AssertionError:
        result = {"memleak": "True"}
    finally:
        return result


def run_config_accuracy(
    config: TorchBenchModelConfig, metrics: List[str], dryrun: bool = False
) -> Dict[str, str]:
    assert metrics == [
        "accuracy"
    ], f"When running accuracy test, others metrics are not supported: {metrics}."
    print(f"Running {config} ...", end="", flush=True)
    if dryrun:
        print(" [skip_by_dryrun]", flush=True)
        return {"accuracy": "skip_by_dryrun"}
    if config.skip:
        print(" [skip]", flush=True)
        return {"accuracy": "skip"}
    try:
        accuracy = get_model_accuracy(config)
        print(" [done]", flush=True)
        return {"accuracy": accuracy}
    except NotImplementedError:
        print(" [not_implemented]", flush=True)
        return {"accuracy": "not_implemented"}
    except OSError as e:
        print(" [oserror]", flush=True)
        return {"accuracy": str(e)}
    except KeyboardInterrupt:
        print(" [user_interrupt]")
        exit(1)
    except Exception as e:
        print(" [runtime_error]", flush=True)
        return {"accuracy": str(e)}


def parse_known_args(args):
    parser = argparse.ArgumentParser()
    default_device = "cuda" if "cuda" in list_devices() else "cpu"
    parser.add_argument(
        "--models",
        "-m",
        default=None,
        help="Name of models to run, split by comma.",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Name of models to skip running, split by comma."
    )
    parser.add_argument(
        "--device",
        "-d",
        default=default_device,
        choices=list_devices(),
        help="Devices to run, splited by comma.",
    )
    parser.add_argument(
        "--test", "-t", default="eval", help="Tests to run, splited by comma."
    )
    parser.add_argument(
        "--bs", default=None, help="Optionally, specify the batch size."
    )
    parser.add_argument(
        "--config", "-c", default=None, help="YAML config to specify tests to run."
    )
    parser.add_argument(
        "--run-bisect", help="Run with the output of regression detector."
    )
    parser.add_argument(
        "--oss",
        action="store_true",
        help="[Meta-Internal Only] Run only the oss models.",
    )
    parser.add_argument(
        "--timm", action="store_true", help="Run with extended timm models."
    )
    parser.add_argument(
        "--huggingface",
        action="store_true",
        help="Run with extended huggingface models.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run with all available models."
    )
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    parser.add_argument(
        "--output",
        default=get_default_output_json_path(BM_NAME),
        help="Specify the path of the output file",
    )
    parser.add_argument("--debug", action="store_true", help="Save the debug output.")
    return parser.parse_known_args(args)


def run(args: List[str]):
    args, extra_args = parse_known_args(args)
    if args.run_bisect:
        configs = generate_model_configs_from_bisect_yaml(args.run_bisect)
    else:
        modelset = set(list_models(internal=(not args.oss)))
        timm_set = set(list_extended_models(suite_name="timm"))
        huggingface_set = set(list_extended_models(suite_name="huggingface"))
        modelset = modelset.union(timm_set).union(huggingface_set)
        if not args.models:
            args.models = []
        if not args.skip:
            args.skip = []
        args.models = parse_str_to_list(args.models)
        args.skip = parse_str_to_list(args.skip)
        if args.timm:
            args.models.extend(timm_set)
        if args.huggingface:
            args.models.extend(huggingface_set)
        if args.all:
            args.models.extend(modelset)
        # If nothing is specified, run all built-in models by default.
        if not args.models:
            args.models = set(list_models(internal=(not args.oss)))

        devices = validate(parse_str_to_list(args.device), list_devices())
        tests = validate(parse_str_to_list(args.test), list_tests())
        batch_sizes = parse_str_to_list(args.bs)
        models = validate(args.models, modelset)
        skips = validate(args.skip, modelset)
        configs = generate_model_configs(
            devices, tests, batch_sizes,
            model_names=models, skip_models=skips, extra_args=extra_args
        )
    debug_output_dir = get_default_debug_output_dir(args.output) if args.debug else None
    configs = (
        init_output_dir(configs, debug_output_dir) if debug_output_dir else configs
    )
    results = {}
    try:
        for config in configs:
            metrics = get_metrics(config)
            if "accuracy" in metrics:
                metrics_dict = run_config_accuracy(config, metrics, dryrun=args.dryrun)
            elif "memleak" in metrics:
                metrics_dict = run_config_memleak(config, dryrun=args.dryrun)
            else:
                metrics_dict = run_config(config, metrics, dryrun=args.dryrun)
            config_str = config_to_str(config)
            for metric in metrics_dict:
                results[f"{config_str}, metric={metric}"] = metrics_dict[metric]
    except KeyboardInterrupt:
        print("User keyboard interrupted!")
    result = get_output_json(BM_NAME, results)
    if args.device == "cuda":
        import torch

        result["environ"]["device"] = torch.cuda.get_device_name()
    o = json.dumps(result, indent=4)
    print(o)
    with open(args.output, "w") as f:
        f.write(o)
