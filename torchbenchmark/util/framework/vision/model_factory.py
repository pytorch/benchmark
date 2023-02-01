import os
import torch
import typing
import torch.optim as optim
import torchvision.models as models
from contextlib import nullcontext
from torchbenchmark.util.model import BenchmarkModel
from typing import Tuple, Generator, Optional

class TorchVisionModel(BenchmarkModel):
    # To recognize this is a torchvision model
    TORCHVISION_MODEL = True
    # These two variables should be defined by subclasses
    DEFAULT_TRAIN_BSIZE = None
    DEFAULT_EVAL_BSIZE = None
    # Default eval precision on CUDA device is fp16
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    def __init__(self, model_name, test, device, jit=False, batch_size=None, weights=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        if weights is None:
            self.model = getattr(models, model_name)(pretrained=True).to(self.device)
        else:
            self.model = getattr(models, model_name)(weights=weights).to(self.device)
        self.example_inputs = (torch.randn((self.batch_size, 3, 224, 224)).to(self.device),)
        if test == "train":
            self.example_outputs = torch.rand_like(self.model(*self.example_inputs))
            self.model.train()
            # setup optimizer and loss_fn
            # if backend is cudagraph, must set optimizer to be capturable
            capturable = bool(int(os.getenv("ADAM_CAPTURABLE", 0))) \
                if not self.opt_args.backend == "cudagraph" else True
            self.opt = optim.Adam(self.model.parameters(), capturable=capturable)
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.real_input = [ torch.rand_like(self.example_inputs[0]) ]
            self.real_output = [ torch.rand_like(self.example_outputs) ]
        elif test == "eval":
            self.model.eval()
            self.example_outputs = torch.rand_like(self.model(*self.example_inputs))
            self.real_input = [ torch.rand_like(self.example_inputs[0]) ]
            self.real_output = [ torch.rand_like(self.example_outputs) ]

        self.amp_context = nullcontext

    def get_flops(self):
        return self.flops, self.batch_size

    def gen_inputs(self, num_batches:int=1) -> Tuple[Generator, Optional[int]]:
        def _gen_inputs():
            while True:
                result = []
                for _i in range(num_batches):
                    result.append((torch.randn((self.batch_size, 3, 224, 224)).to(self.device),))
                if self.dargs.precision == "fp16":
                    result = list(map(lambda x: (x[0].half(), ), result))
                yield result
        return (_gen_inputs(), None)

    def enable_fp16_half(self):
        self.model = self.model.half()
        self.example_inputs = (self.example_inputs[0].half(), )

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        self.opt.zero_grad()
        for data, target in zip(self.real_input, self.real_output):
            with self.amp_context():
                pred = self.model(data)
            self.loss_fn(pred, target).backward()
            self.opt.step()

    def cudagraph_train(self):
        for data, target in zip(self.real_input, self.real_output):
            self.example_inputs[0].copy_(data)
            self.example_outputs.copy_(target)
            self.g.replay()

    def eval(self) -> typing.Tuple[torch.Tensor]:
        model = self.model
        example_inputs = self.example_inputs
        with self.amp_context():
            for data, _target in zip(self.real_input, self.real_output):
                self.example_inputs[0].copy_(data)
                self.example_outputs.copy_(model(*example_inputs))
                break
        return (self.example_outputs, )

    def cudagraph_eval(self):
        for data, target in zip(self.real_input, self.real_output):
            self.example_inputs[0].copy_(data)
            self.example_outputs.copy_(target)
            self.g.replay()
            break
        return (self.example_outputs, )

    def enable_amp(self):
        if not self.dynamo and self.opt_args.cudagraph:
            return NotImplementedError("AMP not implemented for cudagraphs")
        self.amp_context = lambda: torch.cuda.amp.autocast(dtype=torch.float16)
