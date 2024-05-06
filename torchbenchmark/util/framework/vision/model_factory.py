import os
import typing
from contextlib import nullcontext

import torch
import torch.optim as optim
import torchvision.models as models
from torchbenchmark.util.model import BenchmarkModel


class TorchVisionModel(BenchmarkModel):
    # To recognize this is a torchvision model
    TORCHVISION_MODEL = True
    # These two variables should be defined by subclasses
    DEFAULT_TRAIN_BSIZE = None
    DEFAULT_EVAL_BSIZE = None
    # Default eval precision on CUDA device is fp16
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"
    # Whether to skip the opt zero grad
    SKIP_ZERO_GRAD = False

    def __init__(
        self, model_name, test, device, batch_size=None, weights=None, extra_args=[]
    ):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        if weights is None:
            self.model = getattr(models, model_name)(pretrained=True).to(self.device)
        else:
            self.model = getattr(models, model_name)(weights=weights).to(self.device)
        self.example_inputs = (
            torch.randn((self.batch_size, 3, 224, 224)).to(self.device),
        )
        if test == "train":
            # compute loss
            with torch.no_grad():
                self.example_outputs = (
                    torch.rand_like(self.model(*self.example_inputs)),
                )
            self.model.train()
            # setup optimizer and loss_fn
            # if backend is cudagraph, must set optimizer to be capturable
            capturable = (
                bool(int(os.getenv("ADAM_CAPTURABLE", 0)))
                if not (
                    hasattr(self.opt_args, "backend")
                    and self.opt_args.backend == "cudagraph"
                )
                else True
            )
            self.opt = optim.Adam(self.model.parameters(), capturable=capturable)
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif test == "eval":
            self.model.eval()

        self.amp_context = nullcontext
        if hasattr(self.opt_args, "backend") and self.opt_args.backend == "cudagraph":
            self.real_input = (torch.rand_like(self.example_inputs[0]),)
            self.real_output = (torch.rand_like(self.example_outputs),)

    def get_flops(self):
        # By default, FlopCountAnalysis count one fused-mult-add (FMA) as one flop.
        # However, in our context, we count 1 FMA as 2 flops instead of 1.
        # https://github.com/facebookresearch/fvcore/blob/7a0ef0c0839fa0f5e24d2ef7f5d48712f36e7cd7/fvcore/nn/flop_count.py
        assert (
            self.test == "eval"
        ), "fvcore flops is only available on inference tests, as it doesn't measure backward pass."
        from fvcore.nn import FlopCountAnalysis

        FLOPS_FMA = 2.0
        self.flops = FlopCountAnalysis(self.model, tuple(self.example_inputs)).total()
        self.flops = self.flops * FLOPS_FMA
        return self.flops

    def get_input_iter(self):
        """Yield randomized batch size of inputs."""
        import math, random

        n = int(math.log2(self.batch_size))
        buckets = [2**n for n in range(n)]
        while True:
            random_batch_size = random.choice(buckets)
            example_input = (
                torch.randn((random_batch_size, 3, 224, 224)).to(self.device),
            )
            yield example_input

    def get_module(self):
        return self.model, self.example_inputs

    def forward(self):
        with torch.no_grad():
            self.example_outputs = (torch.rand_like(self.model(*self.example_inputs)),)
        for data, target in zip(self.example_inputs, self.example_outputs):
            # Alexnet returns non-grad tensors in forward pass
            # Force to call requires_grad_(True) here
            pred = self.model(data).requires_grad_(True)
            u = self.loss_fn(pred, target)
            return u

    def backward(self, loss):
        loss.backward()

    def optimizer_step(self):
        self.opt.step()

    def cudagraph_train(self):
        for data, target in zip(self.real_input, self.real_output):
            self.example_inputs[0].copy_(data)
            self.example_outputs.copy_(target)
            self.g.replay()

    def eval(self) -> typing.Tuple[torch.Tensor]:
        with self.amp_context():
            return self.model(*self.example_inputs)

    def cudagraph_eval(self):
        for data, target in zip(self.real_input, self.real_output):
            self.example_inputs[0].copy_(data)
            self.example_outputs.copy_(target)
            self.g.replay()
            break
        return (self.example_outputs,)
