"""
Run PyTorch cpu benchmarking.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
import yaml

from datetime import datetime
from pathlib import Path
from typing import List
from .cpu_utils import REPO_PATH, parse_str_to_list, validate, get_output_dir, get_output_json, dump_output, analyze
from ..utils import add_path

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import (list_models, TorchBenchModelConfig,
                                                            list_devices, list_tests)

BM_NAME = "cpu"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def generate_model_configs(devices: List[str], tests: List[str], model_names: List[str], batch_size: int, jit: bool, extra_args: List[str]) -> List[TorchBenchModelConfig]:
    """Use the default batch size and default mode."""
    if not model_names:
        model_names = list_models()
    cfgs = itertools.product(*[devices, tests, model_names])
    result = [TorchBenchModelConfig(
        name=model_name,
        device=device,
        test=test,
        batch_size=batch_size,
        jit=jit,
        extra_args=extra_args,
        extra_env=None,
    ) for device, test, model_name in cfgs]
    return result

def dump_result_to_json(metrics, output_dir, fname):
    result = get_output_json(BM_NAME, metrics)
    dump_output(BM_NAME, result, output_dir, fname)

def generate_model_configs_from_yaml(yaml_file: str) -> List[TorchBenchModelConfig]:
    yaml_file_path = os.path.join(CURRENT_DIR, yaml_file)
    with open(yaml_file_path, "r") as yf:
        config_obj = yaml.safe_load(yf)
    models = config_obj["model"] if "model" in config_obj else None
    models = validate(parse_str_to_list(models), list_models()) if models else list_models()
    extra_args = config_obj["extra_args"].split(' ') if config_obj["extra_args"] else []
    configs = []
    for model in models:
        config = TorchBenchModelConfig(
            name=model,
            device="cpu",
            test=config_obj["test"],
            batch_size=config_obj["batch_size"] if "batch_size" in config_obj else None,
            jit=config_obj["jit"] if "jit" in config_obj else False,
            extra_args=extra_args,
            extra_env=None,
        )
        configs.append(config)
    return configs

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", default="cpu", help="Devices to run, splited by comma.")
    parser.add_argument("--test", "-t", default="eval", help="Tests to run, splited by comma.")
    parser.add_argument("--model", "-m", default=None, help="Only run the specifice models, splited by comma.")
    parser.add_argument("--batch-size", "-b", default=None, help="Run the specifice batch size.")
    parser.add_argument("--jit", action="store_true", help="Convert the models to jit mode.")
    parser.add_argument("--config", "-c", default=None, help="YAML config to specify tests to run.")
    parser.add_argument("--metrics", default="latencies", help="Benchmark metrics, split by comma.")
    parser.add_argument("--output", "-o", default=None, help="Output dir.")
    parser.add_argument("--timeout", default=None, help="Limit single model test run time. Default None, means no limitation.")
    parser.add_argument("--launcher", action="store_true", help="Use torch.backends.xeon.run_cpu to get the peak performance on Intel(R) Xeon(R) Scalable Processors.")
    parser.add_argument("--launcher-args", default="--throughput-mode", help="Provide the args of torch.backends.xeon.run_cpu. See `python -m torch.backends.xeon.run_cpu --help`")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    return parser.parse_known_args(args)

def run(args: List[str]):
    args, extra_args = parse_args(args)
    test_date = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    if args.config:
        configs = generate_model_configs_from_yaml(args.config)
    else:
        # If not specified, use the entire model set
        if not args.model:
            args.model = list_models()
        devices = validate(parse_str_to_list(args.device), list_devices())
        tests = validate(parse_str_to_list(args.test), list_tests())
        models = validate(parse_str_to_list(args.model), list_models())
        configs = generate_model_configs(devices, tests, model_names=models, batch_size=args.batch_size, jit=args.jit, extra_args=extra_args)
    args.output = args.output if args.output else get_output_dir(BM_NAME, test_date)
    try:
        for config in configs:
            run_benchmark(config, args)
    except KeyboardInterrupt:
        print("User keyboard interrupted!")
    result_metrics = analyze(args.output)
    dump_result_to_json(result_metrics, Path(args.output).parent, f"metrics-{test_date}.json")

def run_benchmark(config, args):
    benchmark_script = REPO_PATH.joinpath("userbenchmark", "cpu", "run_config.py")

    cmd = [sys.executable]
    if args.launcher:
        cmd.extend(["-m", "torch.backends.xeon.run_cpu"])
        if args.launcher_args:
            import shlex
            cmd.extend(shlex.split(args.launcher_args))
    cmd.append(str(benchmark_script))
    if config.name:
        cmd.append("-m")
        cmd.append(config.name)
    if config.device:
        cmd.append("-d")
        cmd.append(config.device)
    if config.batch_size:
        cmd.append("-b")
        cmd.append(str(config.batch_size))
    if config.test:
        cmd.append("-t")
        cmd.append(config.test)
    if config.jit:
        cmd.append("--jit")

    cmd.extend(config.extra_args)
    cmd.append("--metrics")
    cmd.append(args.metrics)
    cmd.append("-o")
    cmd.append(str(args.output))

    print(f"\nRunning benchmark: {' '.join(map(str, cmd))}")
    if not args.dryrun:
        timeout = int(args.timeout) if args.timeout else None
        try:
            subprocess.run(cmd, cwd=REPO_PATH, check=False, timeout=timeout)
        except Exception as e:
            print(e)
