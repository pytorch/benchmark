import torch
import random
import numpy as np
import typing
import torch.optim as optim
import torchvision.models as models
from torchbenchmark.util.model import BenchmarkModel

class TorchVisionModel(BenchmarkModel):
    optimized_for_inference = True
    # To recognize this is a torchvision model
    TORCHVISION_MODEL = True
    # These two variables should be defined by subclasses
    DEFAULT_TRAIN_BSIZE = None
    DEFAULT_EVAL_BSIZE = None

    def __init__(self, model_name, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        self.model = getattr(models, model_name)(pretrained=True).to(self.device)
        self.example_inputs = (torch.randn((self.batch_size, 3, 224, 224)).to(self.device),)
        self.example_outputs = torch.rand_like(self.model(*self.example_inputs))
        if test == "train":
            self.model.train()
            # setup optimizer and loss_fn
            self.optimizer = optim.Adam(self.model.parameters())
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif test == "eval":
            self.model.eval()

    # By default, FlopCountAnalysis count one fused-mult-add (FMA) as one flop.
    # However, in our context, we count 1 FMA as 2 flops instead of 1.
    # https://github.com/facebookresearch/fvcore/blob/7a0ef0c0839fa0f5e24d2ef7f5d48712f36e7cd7/fvcore/nn/flop_count.py
    def get_flops(self, flops_fma=2.0):
        if self.test == 'eval':
            flops = self.flops / self.batch_size * flops_fma
            return flops, self.batch_size
        elif self.test == 'train':
            flops = self.flops / self.batch_size * flops_fma
            return flops, self.batch_size
        assert False, f"get_flops() only support eval or train mode, but get {self.test}. Please submit a bug report."

    def enable_fp16_half(self):
        self.model = self.model.half()
        self.example_inputs = (self.example_inputs[0].half(), )

    def get_module(self):
        return self.model, self.example_inputs

    def train(self, niter=3):
        real_input = [ torch.rand_like(self.example_inputs[0]) ]
        real_output = [ torch.rand_like(self.example_outputs) ]
        for _ in range(niter):
            self.optimizer.zero_grad()
            for data, target in zip(real_input, real_output):
                if self.extra_args.cudagraph:
                    self.example_inputs[0].copy_(data)
                    self.example_outputs.copy_(target)
                    self.g.replay()
                else:
                    pred = self.model(data)
                    self.loss_fn(pred, target).backward()
                    self.optimizer.step()

    def eval(self, niter=1) -> typing.Tuple[torch.Tensor]:
        if self.extra_args.cudagraph:
            return NotImplementedError("CUDA Graph is not yet implemented for inference.")
        model = self.model
        example_inputs = self.example_inputs
        result = None
        for _i in range(niter):
            result = model(*example_inputs)
        return (result, )
