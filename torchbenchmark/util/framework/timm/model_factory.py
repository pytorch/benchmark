import typing
from contextlib import suppress

import timm
import torch
from torchbenchmark.util.model import BenchmarkModel

from .extended_configs import BATCH_SIZE_DIVISORS, TIMM_MODELS
from .timm_config import TimmConfig

# No pretrained weights exist for specific TIMM models
DISABLE_PRETRAINED_WEIGHTS = [
    "vovnet39a",
    "vit_giant_patch14_224",
]


class TimmModel(BenchmarkModel):
    # To recognize this is a timm model
    TIMM_MODEL = True
    # These two variables should be defined by subclasses
    DEFAULT_TRAIN_BSIZE = None
    DEFAULT_EVAL_BSIZE = None
    # Default eval precision on CUDA device is fp16
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    def __init__(self, model_name, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        pretrained_weights = (
            True if not model_name in DISABLE_PRETRAINED_WEIGHTS else False
        )
        self.model = timm.create_model(
            model_name,
            in_chans=3,
            scriptable=False,
            num_classes=None,
            drop_rate=0.0,
            drop_path_rate=None,
            drop_block_rate=None,
            pretrained=pretrained_weights,
        )

        self.cfg = TimmConfig(model=self.model, device=device)
        self.example_inputs = self._gen_input(self.batch_size)

        self.model.to(device=self.device)
        if test == "train":
            self.model.train()
        elif test == "eval":
            self.model.eval()
        self.amp_context = suppress

    def get_input_iter(self):
        """Yield randomized batch size of inputs."""
        import math, random

        n = int(math.log2(self.batch_size))
        buckets = [2**n for n in range(n)]
        while True:
            random_batch_size = random.choice(buckets)
            example_input = (self._gen_input(random_batch_size),)
            yield example_input

    def _gen_input(self, batch_size):
        return torch.randn((batch_size,) + self.cfg.input_size, device=self.device)

    def _gen_target(self, batch_size):
        return torch.empty(
            (batch_size,) + self.cfg.target_shape, device=self.device, dtype=torch.long
        ).random_(self.cfg.num_classes)

    def _step_train(self):
        self.cfg.optimizer.zero_grad()
        with self.amp_context():
            output = self.model(self.example_inputs)
        if isinstance(output, tuple):
            output = output[0]
        target = self._gen_target(output.shape[0])
        self.cfg.loss(output, target).backward()
        self.cfg.optimizer.step()

    def _step_eval(self):
        output = self.model(self.example_inputs)
        return output

    def get_optimizer(self):
        return self.cfg.optimizer

    def set_optimizer(self, optimizer) -> None:
        self.cfg.optimizer = optimizer

    def enable_channels_last(self):
        self.model = self.model.to(memory_format=torch.channels_last)
        self.example_inputs = self.example_inputs.contiguous(
            memory_format=torch.channels_last
        )

    def get_module(self):
        return self.model, (self.example_inputs,)

    def train(self):
        self._step_train()

    def eval(self) -> typing.Tuple[torch.Tensor]:
        with self.amp_context():
            out = self._step_eval()
        return (out,)


class ExtendedTimmModel(TimmModel):
    DEFAULT_TRAIN_BSIZE = None
    DEFAULT_EVAL_BSIZE = None
    DEFAULT_CUDA_EVAL_PRECISION_FP32_MODELS = [
        "convit_base",
        "beit_base_patch16_224",
        "xcit_large_24_p8_224",
    ]

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        recorded_batch_size = TIMM_MODELS[self.name]
        if self.name in BATCH_SIZE_DIVISORS:
            recorded_batch_size = max(
                int(recorded_batch_size / BATCH_SIZE_DIVISORS[self.name]), 1
            )
        self.DEFAULT_EVAL_BSIZE = recorded_batch_size
        self.DEFAULT_TRAIN_BSIZE = recorded_batch_size
        self.DEFAULT_EVAL_CUDA_PRECISION = "fp32" \
            if self.name in self.DEFAULT_CUDA_EVAL_PRECISION_FP32_MODELS else "fp16"
        super().__init__(
            model_name=self.name,
            test=test,
            device=device,
            batch_size=batch_size,
            extra_args=extra_args,
        )
