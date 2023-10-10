import os

import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.parallel.style import ColwiseParallel, RowwiseParallel
from torchbenchmark.tasks import NLP

from ...util.model import BenchmarkModel
from .model import LLaMA
from .tp import apply_tp


class Model(BenchmarkModel):
    task = NLP.GENERATION
    DEFAULT_EVAL_BSIZE = 1

    def validate_environment(self):
        if not torch.cuda.is_available() or "cuda" not in self.device:
            return NotImplementedError("Model requires CUDA")

        if not torch.cuda.is_bf16_supported():
            return NotImplementedError("Model requires BF16")

        if not hasattr(self, "_world_size"):
            return NotImplementedError("Model needs to be run via dynamo torchbench and be provided distributed parameters")

        if self._world_size != torch.cuda.device_count():
            return NotImplementedError(
                f"DTensor and all local GPUs to be within the device mesh. {torch.cuda.device_count()} local GPUs, but only world size is only {self._world_size}"
            )

        return None

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test,
            device=device,
            batch_size=batch_size,
            extra_args=extra_args,
        )

        error = self.validate_environment()
        if error:
            raise error

        # temporary workarounds
        torch._inductor.config.allow_buffer_reuse = False
        torch._inductor.config.inplace_buffers = False

        model = LLaMA.from_name("7B")

        print("Applying tensor parallel to model ...")
        apply_tp(model, self._rank, self._world_size)

        max_batch_size = self.batch_size
        with torch.device(device):
            model.setup_caches(
                max_batch_size=max_batch_size, max_seq_length=model.config.block_size
            )

        self.model = model.to(device=device, dtype=torch.bfloat16)

        prompt_size = 10
        idx = torch.randint(
            self.model.config.vocab_size,
            (max_batch_size, prompt_size),
            dtype=torch.int32,
            device=device,
        )
        input_pos = torch.arange(prompt_size, device=device)
        self.example_inputs = [idx, input_pos]

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        raise NotImplementedError("Training not supported for this model")

    def eval(self):
        raise NotImplementedError("Model needs to be run via dynamo torchbench and be provided distributed parameters")
