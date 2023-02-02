import os
import pathlib
import sys
import time
from typing import List, Optional, Tuple
import torch
import re
from run_sweep import WORKER_TIMEOUT, WARMUP_ROUNDS, ModelTestResult, NANOSECONDS_PER_MILLISECONDS
from torchbenchmark import ModelTask
import numpy


def parse_func(func):
    description = str(func)
    reg_method = re.compile(r"method (.*) of (.*) object")
    reg_method2 = re.compile(r"wrapper (.*) of (.*) object")
    reg_function = re.compile(r"function (.*)[ >]")
    reg_class = re.compile(r"class (.*)[ >]")
    reg_generator = re.compile(r"torch._C.Generator object at (.*)")
    result_method = reg_method.findall(description)
    result_function = reg_function.findall(description)
    result_method2 = reg_method2.findall(description)
    result_class = reg_class.findall(description)
    result_generator = reg_generator.findall(description)
    if result_method:
        func_name = result_method[0][0]
        module_name = result_method[0][1]
    elif result_function:
        func_name = result_function[0].split("at 0x")[0].strip()
        module_name = ''
    elif result_method2:
        func_name = result_method2[0][0]
        module_name = result_method2[0][1]
    elif result_class:
        func_name = result_class[0].split("at 0x")[0].strip()
        module_name = ''
    elif result_generator:
        func_name = 'Generator'
        module_name = 'torch._C'
    else:
        # check if the func has attribute `__module__` and `__name__`
        if hasattr(func, '__module__'):
            module_name = func.__module__
        else:
            module_name = ''
        if hasattr(func, '__name__'):
            func_name = func.__name__
        else:
            func_name = ''
        if module_name != 'torch._ops.profiler':
            print("not match: ", description)
    module_name = module_name.replace("'", "")
    func_name = func_name.replace("'", "")
    return module_name, func_name


def generate_API_list():
    tmp_api_list = set()
    tmpb = set(
        [_ for _ in torch.overrides.get_ignored_functions() if _ not in [True, False]])
    tmpa = set(torch.overrides.get_testing_overrides().keys())
    raw_all_apis = tmpa.union(tmpb)
    # collect all items' attribute  `module` to a list
    for item in raw_all_apis:
        module_name, func_name = parse_func(item)
        # if (module_name, func_name) in api_list:
        # print("duplicated: ", (module_name, func_name))
        tmp_api_list.add((module_name, func_name))
    return tmp_api_list

API_LIST = generate_API_list()


class CoverageMode(torch.overrides.TorchFunctionMode):

    def __init__(self, model='', output_file=None):
        self.model = model
        self.seen = set()
        self.api_used = set()
        self.output_file = output_file

    def check_func_in_APIs(self, func):
        module_name, func_name = parse_func(func)
        if (module_name, func_name) not in API_LIST and module_name != 'torch._ops.profiler':
            print("not in APIs: (%s, %s)" % (module_name, func_name))
        else:
            self.api_used.add((module_name, func_name))
            # debug
            # print("in APIs: ", (module_name, func_name))

    def get_api_coverage_rate(self):
        return len(self.api_used) / len(API_LIST)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.seen.add(func)
        if kwargs is None:
            kwargs = {}
        self.check_func_in_APIs(func)
        return func(*args, **kwargs)

    def commit(self):
        if self.output_file:
            with open(self.output_file, 'a') as f:
                for api in self.api_used:
                    f.write("%s,%s\n" % (api[0], api[1]))


def run_one_step(model, func, device: str, nwarmup=WARMUP_ROUNDS, num_iter=10) -> Tuple[float, Optional[Tuple[torch.Tensor]]]:
    "Run one step of the model, and return the latency in milliseconds."
    # Warm-up `nwarmup` rounds
    for _i in range(nwarmup):
        func()
    result_summary = []
    for _i in range(num_iter):
        if device == "cuda":
            torch.cuda.synchronize()
            # Collect time_ns() instead of time() which does not provide better precision than 1
            # second according to https://docs.python.org/3/library/time.html#time.time.
            t0 = time.time_ns()
            with CoverageMode(model, '/tmp/api_used.csv') as coverage:
                try:
                    func()
                finally:
                    coverage.commit()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            t1 = time.time_ns()
        else:
            t0 = time.time_ns()
            with CoverageMode(model, '/tmp/api_used.csv') as coverage:
                try:
                    func()
                finally:
                    coverage.commit()
            t1 = time.time_ns()
        result_summary.append((t1 - t0) / NANOSECONDS_PER_MILLISECONDS)
    wall_latency = numpy.median(result_summary)
    return wall_latency


def _run_model_test_coverage(model_path: pathlib.Path, test: str, device: str, jit: bool, batch_size: Optional[int], extra_args: List[str]) :
    assert test == "train" or test == "eval", f"Test must be either 'train' or 'eval', but get {test}."
    result = ModelTestResult(name=model_path.name, test=test, device=device, extra_args=extra_args, batch_size=None, precision="fp32",
                             status="OK", results={})

    # Run the benchmark test in a separate process
    print(f"Running model {model_path.name} ... ", end='', flush=True)
    status: str = "OK"
    bs_name = "batch_size"
    correctness_name = "correctness"
    error_message: Optional[str] = None
    try:
        task = ModelTask(os.path.basename(model_path), timeout=WORKER_TIMEOUT)
        if not task.model_details.exists:
            status = "NotExist"
            return
        task.make_model_instance(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        # Check the batch size in the model matches the specified value
        result.batch_size = task.get_model_attribute(bs_name)
        result.precision = task.get_model_attribute("dargs", "precision")
        if batch_size and (not result.batch_size == batch_size):
            raise ValueError(
                f"User specify batch size {batch_size}, but model {result.name} runs with batch size {result.batch_size}. Please report a bug.")
        result.results["latency_ms"] = run_one_step(model_path.name, task.invoke, device)
        # if NUM_BATCHES is set, update to per-batch latencies
        num_batches = task.get_model_attribute("NUM_BATCHES")
        if num_batches:
            result.results["latency_ms"] = result.results["latency_ms"] / num_batches
        # if the model provides eager eval result, save it for cosine similarity
        correctness = task.get_model_attribute(correctness_name)
        if correctness is not None:
            result.results[correctness_name] = str(correctness)
    except NotImplementedError as e:
        status = "NotImplemented"
        error_message = str(e)
    except TypeError as e:  # TypeError is raised when the model doesn't support variable batch sizes
        status = "TypeError"
        error_message = str(e)
    except KeyboardInterrupt as e:
        status = "UserInterrupted"
        error_message = str(e)
    except Exception as e:
        status = f"{type(e).__name__}"
        error_message = str(e)
    finally:
        print(f"[ {status} ]")
        result.status = status
        if error_message:
            result.results["error_message"] = error_message
        if status == "UserInterrupted":
            sys.exit(1)
        return result
