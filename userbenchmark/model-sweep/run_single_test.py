import pathlib
import sys
from typing import Optional, List

def _run_model_test(model_path: pathlib.Path, test: str, device: str, jit: bool, batch_size: Optional[int], extra_args: List[str]) -> ModelTestResult:
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
            raise ValueError(f"User specify batch size {batch_size}, but model {result.name} runs with batch size {result.batch_size}. Please report a bug.")
        result.results["latency_ms"] = run_one_step(task.invoke, device)
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
        return result