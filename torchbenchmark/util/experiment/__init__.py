"""
Utilities to run experiments across models.
Utility functions in this file don't handle exceptions.
They expect their callers to handle exceptions themselves.
"""
import os
import sys
import torch
import time
import dataclasses
from typing import Optional, List, Dict
from torchbenchmark import ModelTask

WORKER_TIMEOUT = 600 # seconds
BS_FIELD_NAME = "batch_size"

def get_model_task(name: str, timeout: float=WORKER_TIMEOUT, extra_env: Optional[Dict[str, str]]=None) -> ModelTask:
    """ Get model constructor in a subprocess. """
    task = ModelTask(name, timeout=timeout, extra_env=extra_env)
    if not task.model_details.exists:
        raise ValueError("Failed to import model task: {name}. Please run the model manually to make sure it succeeds, or report a bug.")
    return task

def create_model_instance(task: ModelTask, device: str, test: str, batch_size: str,
                          jit: bool, extra_args: List[str]):
    task.make_model_instance(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
    task_batch_size = task.get_model_attribute(BS_FIELD_NAME)
    if batch_size and (not batch_size == task_batch_size):
        raise ValueError(f"User specify batch size {batch_size}," +
                         f"but model {task.name} runs with batch size {task_batch_size}. Please report a bug.")
