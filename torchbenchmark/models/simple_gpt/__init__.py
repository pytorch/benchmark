import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.parallel.style import ColwiseParallel, RowwiseParallel
from torchbenchmark.tasks import NLP

from ...util.model import BenchmarkModel
from .model import LLaMA


class Model(BenchmarkModel):
    task = NLP.GENERATION
    DEFAULT_EVAL_BSIZE = 1

    def validate_environment(self):
        if not torch.cuda.is_available() or "cuda" not in self.device:
            return NotImplementedError("Model requires CUDA")

        if not torch.cuda.is_bf16_supported():
            return NotImplementedError("Model requires BF16")

        if not hasattr(self, "_world_size"):
            return NotImplementedError(
                "Model needs to be run via dynamo torchbench and be provided distributed parameters"
            )

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

        self.model = LLaMA.from_name("7B", self._world_size).to(
            device=device, dtype=torch.bfloat16
        )

        # Tensor parallelism using DTensor
        mesh = DeviceMesh("cuda", list(range(self._world_size)))
        for block in self.model.transformer.h:
            # prepare attention weights to be parallelized
            block.attn.prepare_qkv_for_dtensor_tp()

            parallelize_module(
                module=block,
                device_mesh=mesh,
                parallelize_plan={
                    "attn.c_attn_q": ColwiseParallel(),
                    "attn.c_attn_k": ColwiseParallel(),
                    "attn.c_attn_v": ColwiseParallel(),
                    "attn.c_proj": RowwiseParallel(),
                    "mlp.c_fc1": ColwiseParallel(),
                    "mlp.c_fc2": ColwiseParallel(),
                    "mlp.c_proj": RowwiseParallel(),
                },
                tp_mesh_dim=0,
            )

        max_batch_size = self.batch_size
        self.model.setup_caches(
            max_batch_size=max_batch_size, max_seq_length=self.model.config.block_size
        )

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
        raise NotImplementedError(
            "Model needs to be run via dynamo torchbench and be provided distributed parameters"
        )
