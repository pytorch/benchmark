import importlib
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from torchbenchmark.models import (hf_Bert, hf_BertLarge, hf_GPT2,
                                   hf_GPT2_large, hf_T5, hf_T5_large, resnet50,
                                   resnet152, timm_vision_transformer,
                                   timm_vision_transformer_large)
from torchbenchmark.util.model import BenchmarkModel


def get_benchmark_model(
    model_name: str,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = torch.device("cuda"),
) -> Tuple[nn.Module, Callable, optim.Optimizer, Any]:
    pos = model_name.rfind(".")
    module = importlib.import_module(model_name[:pos])
    model_class = getattr(module, model_name[(pos + 1) :])
    model: BenchmarkModel = model_class("train", device, batch_size=batch_size)

    nn_model = model.model
    example_inputs = model.example_inputs
    model_type = type(model)

    if model_type in (resnet50.Model, resnet152.Model):
        example_outputs = model.example_outputs
        optimizer = model.optimizer
        loss_fn = model.loss_fn
        forward = resnet_forward_wrapper(example_outputs, loss_fn)

    elif model_type in (
        hf_T5.Model,
        hf_GPT2.Model,
        hf_Bert.Model,
        hf_BertLarge.Model,
        hf_T5_large.Model,
        hf_GPT2_large.Model,
    ):
        forward = hf_forward_wrapper()
        optimizer = model.optimizer

    elif model_type in [
        timm_vision_transformer_large.Model,
        timm_vision_transformer.Model,
    ]:
        optimizer = model.cfg.optimizer
        loss_fn = model.cfg.loss
        _gen_target = model._gen_target
        amp_context = model.amp_context
        forward = timm_vit_forward_wrapper(loss_fn, amp_context, _gen_target)

    return (nn_model, forward, optimizer, example_inputs)


def resnet_forward_wrapper(example_outputs, loss_fn) -> Callable:
    def resnet_forward(model, example_inputs):
        nonlocal example_outputs, loss_fn
        out = model(*example_inputs)
        loss = loss_fn(out, example_outputs)
        return loss

    return resnet_forward


def hf_forward_wrapper() -> Callable:
    def hf_forward(model, example_inputs):
        loss = model(**example_inputs).loss
        return loss

    return hf_forward


def timm_vit_forward_wrapper(loss_fn, amp_context, _gen_target) -> Callable:
    def timm_vit_forward(model, example_inputs):
        nonlocal loss_fn, amp_context, _gen_target
        with amp_context():
            output = model(example_inputs)
        if isinstance(output, tuple):
            output = output[0]
        target = _gen_target(output.shape[0])
        loss = loss_fn(output, target)
        return loss

    return timm_vit_forward
