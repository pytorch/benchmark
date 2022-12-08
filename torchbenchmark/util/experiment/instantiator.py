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
from torchbenchmark import _list_model_paths, ModelTask

WORKER_TIMEOUT = 600 # seconds
BS_FIELD_NAME = "batch_size"

@dataclasses.dataclass
class TorchBenchModelConfig:
    name: str
    device: str
    test: str
    batch_size: Optional[int]
    jit: bool
    extra_args: List[str]
    extra_env: Optional[Dict[str, str]] = None

def _set_extra_env(extra_env):
    if not extra_env:
        return
    for env_key in extra_env:
        os.environ[env_key] = extra_env[env_key]

def load_model_isolated(config: TorchBenchModelConfig, timeout: float=WORKER_TIMEOUT) -> ModelTask:
    """ Load and return the model in a subprocess. """
    task = ModelTask(config.name, timeout=timeout, extra_env=config.extra_env)
    if not task.model_details.exists:
        raise ValueError(f"Failed to import model task: {config.name}. Please run the model manually to make sure it succeeds, or report a bug.")
    task.make_model_instance(test=config.test, device=config.device, jit=config.jit, batch_size=config.batch_size, extra_args=config.extra_args)
    task_batch_size = task.get_model_attribute(BS_FIELD_NAME)
    # check batch size
    if config.batch_size and (not config.batch_size == task_batch_size):
        raise ValueError(f"User specify batch size {config.batch_size}," +
                         f"but model {task.name} runs with batch size {task_batch_size}. Please report a bug.")
    return task

def load_model(config: TorchBenchModelConfig) -> BenchmarkModel:
    """Load and return a model instance in the same process. """
    package = "torchbenchmark"
    module = importlib.import_module(f'.models.{config.name}', package=package)
    Model = getattr(module, 'Model', None)
    if not Model:
        raise ValueError(f"Error: {module} does not define attribute Model.")
    model_instance = Model(test=config.test, device=config.device, batch_size=config.batch_size, jit=config.jit, extra_args=config.extra_args)
    # check name
    if not model_instance.name == config.name:
        raise ValueError(f"Required model {config.name}, loaded {model_instance.name}.")
    # check batch size
    if config.batch_size and (not config.batch_size == model_instance.batch_size):
        raise ValueError(f"User specify batch size {config.batch_size}," +
                         f"but model {model_instance.name} runs with batch size {model_instance.batch_size}. Please report a bug.")
    _set_extra_env(config.extra_env)
    return model_instance

def list_models() -> List[str]:
    """Return a list of names of all TorchBench models"""
    model_paths = _list_model_paths()
    model_names = list(map(lambda x: os.path.basename(x), model_paths))
    return model_names
