import pathlib
import torch
from typing import Optional, List, Tuple
from torchbenchmark import ModelTask
import os
import sys
import time
import numpy
from components.model_analyzer.TorchBenchAnalyzer import ModelAnalyzer
from run_sweep import WORKER_TIMEOUT, WARMUP_ROUNDS, ModelTestResult, NANOSECONDS_PER_MILLISECONDS


def run_one_step_flops(func, device: str, nwarmup=WARMUP_ROUNDS, num_iter=10, flops=True) -> Tuple[float, float, Optional[Tuple[torch.Tensor]]]:
    "Run one step of the model, and return the latency in milliseconds."
    # Warm-up `nwarmup` rounds
    for _i in range(nwarmup):
        func()
    result_summary = []
    if flops:
        model_analyzer = ModelAnalyzer()
        model_analyzer.start_monitor()
    for _i in range(num_iter):
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
        result_summary.append((t1 - t0) / NANOSECONDS_PER_MILLISECONDS)
    if flops:
        model_analyzer.stop_monitor()
        model_analyzer.aggregate()
        tflops = model_analyzer.calculate_flops()
        
    wall_latency = numpy.median(result_summary)
    return (wall_latency, tflops)

def _run_model_test_proper_bs(model_path: pathlib.Path, test: str, device: str, jit: bool, batch_size: Optional[int], extra_args: List[str]) -> ModelTestResult:
    assert test == "train" or test == "eval", f"Test must be either 'train' or 'eval', but get {test}."
    result = ModelTestResult(name=model_path.name, test=test, device=device, extra_args=extra_args, batch_size=None, precision="fp32",
                             status="OK", results={})

    # Run the benchmark test in a separate process
    print(f"Running model {model_path.name} ... ", flush=True)
    status: str = "OK"
    bs_name = "batch_size"
    correctness_name = "correctness"
    error_message: Optional[str] = None
    result.results['details'] = []
    task = ModelTask(os.path.basename(model_path), timeout=WORKER_TIMEOUT)
    for batch_size_exp in range(8):
        batch_size = 2 ** batch_size_exp
        try:
            print(f"Batch Size {batch_size} ", end='')
            latency_ms_cur = 0
            if not task.model_details.exists:
                status = "NotExist"
                return
            task.make_model_instance(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
            result.precision = task.get_model_attribute("dargs", "precision")
            # Check the batch size in the model matches the specified value
            if batch_size and (not task.get_model_attribute(bs_name) == batch_size):
                raise ValueError(f"User specify batch size {batch_size}, but model {result.name} runs with batch size {task.get_model_attribute(bs_name)}. Please report a bug.")
            latency_ms_cur, tflops_cur = run_one_step_flops(task.invoke, device)
            latency_ms_cur = latency_ms_cur / batch_size
            result.results['details'].append({'batch_size': batch_size, "latency_ms": latency_ms_cur, "tflops": tflops_cur})
            # if the model provides eager eval result, save it for cosine similarity
            correctness = task.get_model_attribute(correctness_name)
            if correctness is not None:
                result.results[correctness_name] = correctness
        except NotImplementedError as e:
            status = "NotImplemented"
            error_message = str(e)
        except TypeError as e: # TypeError is raised when the model doesn't support variable batch sizes
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
            if status != 'OK':
                return result
    # find the best case
    result.results['optimal_latency_bs'] = min(result.results['details'], key=lambda x:x['latency_ms'])['batch_size']
    result.results['optimal_tflops_bs'] = max(result.results['details'], key=lambda x:x['tflops'])['batch_size']
    return result