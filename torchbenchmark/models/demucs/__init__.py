import json
import torch
import random
import numpy as np
from fractions import Fraction

from .demucs.model import Demucs
from .demucs.parser import get_name, get_parser
from .demucs.augment import FlipChannels, FlipSign, Remix, Shift
from .demucs.utils import capture_init, center_trim
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER
from torch import Tensor
from torch.nn.modules.container import Sequential
from torchbenchmark.models.demucs.demucs.model import Demucs
from typing import Optional, Tuple


torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


class DemucsWrapper(torch.nn.Module):
    def __init__(self, model: Demucs, augment: Sequential) -> None:
        super(DemucsWrapper, self).__init__()
        self.model = model
        self.augment = augment

    def forward(self, streams) -> Tuple[Tensor, Tensor]:
        sources = streams[:, 1:]
        sources = self.augment(sources)
        mix = sources.sum(dim=1)
        return sources, self.model(mix)


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS
    # Original train batch size: 64
    # Source: https://github.com/facebookresearch/demucs/blob/3e5ea549ba921316c587e5f03c0afc0be47a0ced/conf/config.yaml#L37
    def __init__(self, test="eval", device: Optional[str]=None, jit: bool=False, train_bs=64, eval_bs=8) -> None:
        super().__init__()
        self.device = device
        self.jit = jit
        self.parser = get_parser()
        self.args = self.parser.parse_args([])
        args = self.args
        self.model = Demucs(channels=64)
        self.dmodel = self.model
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        if 1:
            samples = 80000
            # TODO: enable GPU training after it is supported by infra
            #       see GH issue https://github.com/pytorch/benchmark/issues/652
            # self.example_inputs = (torch.rand([train_bs, 5, 2, 426888], device=device),)
            self.eval_example_inputs = (torch.rand([eval_bs, 5, 2, 426888], device=device),)

        self.duration = Fraction(samples + args.data_stride, args.samplerate)
        self.stride = Fraction(args.data_stride, args.samplerate)

        if args.mse:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.L1Loss()

        if args.augment:
            self.augment = torch.nn.Sequential(FlipSign(), FlipChannels(), Shift(args.data_stride),
                                    Remix(group_size=args.remix_group_size)).to(device)
        else:
            self.augment = Shift(args.data_stride)

        self.model = DemucsWrapper(self.model, self.augment)

        if self.jit:
            if hasattr(torch.jit, '_script_pdt'):
                self.model = torch.jit._script_pdt(self.model, example_inputs = [self.eval_example_inputs, ])
            else:
                self.model = torch.jit.script(self.model, example_inputs = [self.eval_example_inputs, ])

    def _set_mode(self, train):
        self.model.train(train)

    def get_module(self) -> Tuple[DemucsWrapper, Tuple[Tensor]]:
        self.model.eval()
        return self.model, self.eval_example_inputs

    def eval(self, niter=1):
        for _ in range(niter):
            sources, estimates = self.model(*self.eval_example_inputs)
            sources = center_trim(sources, estimates)
            loss = self.criterion(estimates, sources)

    def train(self, niter=1):
        if self.device == "cpu":
            raise NotImplementedError("Disable CPU training because it is too slow (> 1min)")
        if self.device == "cuda":
            raise NotImplementedError("Disable GPU training because it causes CUDA OOM on T4")
        for _ in range(niter):
            sources, estimates = self.model(*self.eval_example_inputs)
            sources = center_trim(sources, estimates)
            loss = self.criterion(estimates, sources)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
