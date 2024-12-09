"""
Utilities to instantiate TorchBench models in the same process or child process.
Functions in this file don't handle exceptions.
They expect callers handle all exceptions.
"""

import dataclasses
import os
import pathlib
from typing import Dict, List, Optional

from torchbenchmark import (
    _list_model_paths,
    load_canary_model_by_name,
    load_model_by_name,
    ModelNotFoundError,
    ModelTask,
)
from torchbenchmark.util.model import BenchmarkModel

WORKER_TIMEOUT = 3600  # seconds
BS_FIELD_NAME = "batch_size"


@dataclasses.dataclass
class TorchBenchModelConfig:
    name: str
    test: str
    device: str
    batch_size: Optional[int]
    extra_args: List[str]
    extra_env: Optional[Dict[str, str]] = None
    output_dir: Optional[pathlib.Path] = None
    skip: bool = False


def _set_extra_env(extra_env):
    if not extra_env:
        return
    for env_key in extra_env:
        os.environ[env_key] = extra_env[env_key]


def inject_model_invoke(model_task: ModelTask, inject_function):
    model_task.replace_invoke(inject_function.__module__, inject_function.__name__)


def load_model_isolated(
    config: TorchBenchModelConfig, timeout: float = WORKER_TIMEOUT
) -> ModelTask:
    """Load and return the model in a subprocess. Optionally, save its stdout and stderr to the specified directory."""
    task = ModelTask(
        config.name,
        timeout=timeout,
        extra_env=config.extra_env,
        save_output_dir=config.output_dir,
    )
    if not task.model_details.exists:
        raise ValueError(
            f"Failed to import model task: {config.name}. Please run the model manually to make sure it succeeds, or report a bug."
        )
    task.make_model_instance(
        test=config.test,
        device=config.device,
        batch_size=config.batch_size,
        extra_args=config.extra_args,
    )
    task_batch_size = task.get_model_attribute(BS_FIELD_NAME)
    # check batch size if not measuring accuracy
    if (
        config.batch_size
        and (not config.batch_size == task_batch_size)
        and not task.get_model_attribute("accuracy")
    ):
        raise ValueError(
            f"User specify batch size {config.batch_size},"
            + f"but model {task.name} runs with batch size {task_batch_size}. Please report a bug."
        )
    return task


def load_model(config: TorchBenchModelConfig) -> BenchmarkModel:
    """Load and return a model instance in the same process."""
    Model = None
    try:
        Model = load_model_by_name(config.name)
    except ModelNotFoundError:
        print(f"Warning: The model {config.name} cannot be found at core set.")
    if not Model:
        try:
            Model = load_canary_model_by_name(config.name)
        except ModelNotFoundError:
            print(
                f"Error: The model {config.name} cannot be found at either core or canary model set."
            )
            exit(-1)

    model_instance = Model(
        test=config.test,
        device=config.device,
        batch_size=config.batch_size,
        extra_args=config.extra_args,
    )
    # check name
    if not model_instance.name == config.name:
        raise ValueError(f"Required model {config.name}, loaded {model_instance.name}.")
    # check batch size if not measuring accuracy
    if (
        config.batch_size
        and (not config.batch_size == model_instance.batch_size)
        and not model_instance.dargs.accuracy
    ):
        raise ValueError(
            f"User specify batch size {config.batch_size},"
            + f"but model {model_instance.name} runs with batch size {model_instance.batch_size}. Please report a bug."
        )
    _set_extra_env(config.extra_env)
    return model_instance


def list_devices() -> List[str]:
    """Return a list of available devices."""
    devices = ["cpu"]
    import torch

    device_type = torch._C._get_accelerator().type
    if device_type != "cpu":
        devices.append(device_type)
    return devices


def list_tests() -> List[str]:
    """Return a list of available tests."""
    return ["train", "eval"]


def list_models(internal=True) -> List[str]:
    """Return a list of names of all TorchBench models"""
    model_paths = _list_model_paths(internal=internal)
    model_names = list(map(lambda x: os.path.basename(x), model_paths))
    return model_names


def list_extended_models(suite_name: str = "all") -> List[str]:
    from torchbenchmark.util.framework.huggingface.list_extended_configs import (
        list_extended_huggingface_models,
    )
    from torchbenchmark.util.framework.timm.extended_configs import (
        list_extended_timm_models,
    )

    if suite_name == "huggingface":
        return list_extended_huggingface_models()
    elif suite_name == "timm":
        return list_extended_timm_models()
    elif suite_name == "all":
        return list_extended_huggingface_models() + list_extended_timm_models()
    else:
        assert (
            False
        ), "Currently, we only support extended model set huggingface or timm."
