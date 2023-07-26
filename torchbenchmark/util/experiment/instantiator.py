"""
Utilities to instantiate TorchBench models in the same process or child process.
Functions in this file don't handle exceptions.
They expect callers handle all exceptions.
"""
import os
import importlib
import dataclasses
from typing import Optional, List, Dict
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark import _list_model_paths, load_model_by_name, ModelTask

WORKER_TIMEOUT = 3600 # seconds
BS_FIELD_NAME = "batch_size"

@dataclasses.dataclass
class TorchBenchModelConfig:
    name: str
    test: str
    device: str
    batch_size: Optional[int]
    extra_args: List[str]
    extra_env: Optional[Dict[str, str]] = None

def _set_extra_env(extra_env):
    if not extra_env:
        return
    for env_key in extra_env:
        os.environ[env_key] = extra_env[env_key]

def inject_model_invoke(model_task: ModelTask, inject_function):
    model_task.replace_invoke(inject_function.__module__, inject_function.__name__)

def load_model_isolated(config: TorchBenchModelConfig, timeout: float=WORKER_TIMEOUT) -> ModelTask:
    """ Load and return the model in a subprocess. """
    task = ModelTask(config.name, timeout=timeout, extra_env=config.extra_env)
    if not task.model_details.exists:
        raise ValueError(f"Failed to import model task: {config.name}. Please run the model manually to make sure it succeeds, or report a bug.")
    task.make_model_instance(test=config.test, device=config.device, batch_size=config.batch_size, extra_args=config.extra_args)
    task_batch_size = task.get_model_attribute(BS_FIELD_NAME)
    # check batch size if not measuring accuracy
    if config.batch_size and (not config.batch_size == task_batch_size) and not task.get_model_attribute('accuracy'):
        raise ValueError(f"User specify batch size {config.batch_size}," +
                         f"but model {task.name} runs with batch size {task_batch_size}. Please report a bug.")
    return task

def load_model(config: TorchBenchModelConfig) -> BenchmarkModel:
    """Load and return a model instance in the same process. """
    Model = load_model_by_name(config.name)
    model_instance = Model(test=config.test, device=config.device, batch_size=config.batch_size, extra_args=config.extra_args)
    # check name
    if not model_instance.name == config.name:
        raise ValueError(f"Required model {config.name}, loaded {model_instance.name}.")
    # check batch size if not measuring accuracy
    if config.batch_size and (not config.batch_size == model_instance.batch_size) and not model_instance.dargs.accuracy:
        raise ValueError(f"User specify batch size {config.batch_size}," +
                         f"but model {model_instance.name} runs with batch size {model_instance.batch_size}. Please report a bug.")
    _set_extra_env(config.extra_env)
    return model_instance

def list_devices() -> List[str]:
    """Return a list of available devices."""
    devices = ["cpu"]
    import torch
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices

def list_tests() -> List[str]:
    """Return a list of available tests."""
    return ["train", "eval"]

def list_models() -> List[str]:
    """Return a list of names of all TorchBench models"""
    model_paths = _list_model_paths()
    model_names = list(map(lambda x: os.path.basename(x), model_paths))
    return model_names
