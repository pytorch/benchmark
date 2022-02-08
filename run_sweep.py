"""
Run a config of benchmarking with a list of models.
If unspecified, run a sweep of all models.
"""
import argparse
import json
import os
import torch
import time
import pathlib
from dataclasses import dataclass
import itertools
from typing import List, Optional
from torchbenchmark import ModelTask

WARMUP_ROUNDS = 3
MODEL_DIR = ['torchbenchmark', 'models']
NANOSECONDS_PER_MILLISECONDS = 1_000_000.0

def run_one_step(func, device, nwarmup=WARMUP_ROUNDS) -> float:
    "Run one step of the model, and return the latency in milliseconds."
    # Warm-up `nwarmup` rounds
    for _i in range(nwarmup):
        func()
    if device == "cuda":
        torch.cuda.synchronize()
        # Collect time_ns() instead of time() which does not provide better precision than 1
        # second according to https://docs.python.org/3/library/time.html#time.time.
        t0 = time.time_ns()
        func()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        t1 = time.time_ns()
    else:
        t0 = time.time_ns()
        func()
        t1 = time.time_ns()
    wall_latency = (t1 - t0) / NANOSECONDS_PER_MILLISECONDS
    return wall_latency

@dataclass
class ModelTestResult:
    name: str
    test: str
    device: str
    extra_args: List[str]
    batch_size: Optional[int]
    latency: Optional[float]
    stdout: Optional[str]
    stderr: Optional[str]

def _list_model_paths(models: List[str]) -> List[str]:
    p = pathlib.Path(__file__).parent.joinpath(*MODEL_DIR)
    model_paths = sorted(child for child in p.iterdir() if child.is_dir())
    valid_model_paths = sorted(filter(lambda x: x.joinpath("__init__.py").exists(), model_paths))
    if models:
        valid_model_paths = sorted(filter(lambda x: x.name in models, valid_model_paths))
    return valid_model_paths

def _validate_tests(tests: str) -> List[str]:
    tests_list = list(map(lambda x: x.strip(), tests.split(",")))
    valid_tests = ['train', 'eval']
    for t in tests_list:
        if t not in valid_tests:
            raise ValueError(f'Invalid test {t} passed into --tests. Expected tests: {valid_tests}.')
    return tests_list

def _validate_devices(devices: str) -> List[str]:
    devices_list = list(map(lambda x: x.strip(), devices.split(",")))
    valid_devices = ['cpu', 'cuda']
    for d in devices_list:
        if d not in valid_devices:
            raise ValueError(f'Invalid device {d} passed into --devices. Expected devices: {valid_devices}.')
    return devices_list

def _run_model_test(model_path: pathlib.Path, test: str, device: str, jit: bool, batch_size: int, extra_args: List[str]) -> ModelTestResult:
    result = ModelTestResult(name=model_path.name, test=test, device=device, extra_args=extra_args,
                             latency=None, stdout=None, stderr=None)
    # Run the benchmark test in a separate process
    print(f"Running model {model_path.name} ... ", end='', flush=True)
    try:
        task = ModelTask(os.path.basename(model_path))
        if not task.model_details.exists:
            result.latency = None
            result.stdout = None
            result.stderr = f"Model {model_path.name} does not exist."
            return
        # TODO: Handle batch size
        task.make_model_instance(test=test, device=device, jit=jit, extra_args=extra_args)
        func = getattr(task, test)
        result.latency = run_one_step(func, device)
        print("[ OK ]")
    except NotImplementedError:
        result.stderr = "Not Implemented"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", nargs='+', default=[],
                        help="Specify one or more models to run. If not set, trigger a sweep-run on all models.")
    parser.add_argument("-t", "--tests", required=True, type=_validate_tests, help="Specify tests, choice of train, or eval.")
    parser.add_argument("-d", "--devices", required=True, type=_validate_devices, help="Specify devices, choice of cpu, or cuda.")
    parser.add_argument("-b", "--bs", type=int, help="Specify batch size.")
    parser.add_argument("--jit", action='store_true', help="Turn on torchscript.")
    parser.add_argument("-o", "--output", type=str, help="The default output json file.")
    args, extra_args = parser.parse_known_args()
    args.models = _list_model_paths(args.models)
    results = []
    for element in itertools.product(*[args.models, args.tests, args.devices]):
        model_path, test, device = element

        r = _run_model_test(model_path, test, device, args.jit, batch_size=args.bs, extra_args=extra_args)
        results.append(r)
