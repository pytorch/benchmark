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
    DEFAULT_TRAIN_BSIZE = 64
    DEFAULT_EVAL_BSIZE = 8

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]) -> None:
        # reduce the eval batch size when running on CPU
        # see: https://github.com/pytorch/benchmark/issues/895
        if device == "cpu":
            self.DEFAULT_EVAL_BSIZE = max(1, int(self.DEFAULT_EVAL_BSIZE / 8))
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.parser = get_parser()
        self.args = self.parser.parse_args([])
        args = self.args
        model = Demucs(channels=64)
        model.to(device)
        samples = 80000

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

        self.model = DemucsWrapper(model, self.augment)
        
        if test == "train":
            self.model.train()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        elif test == "eval":
            self.model.eval()

        self.example_inputs = (torch.rand([self.batch_size, 5, 2, 426888], device=device),)

    def get_module(self) -> Tuple[DemucsWrapper, Tuple[Tensor]]:
        return self.model, self.example_inputs

    def eval(self) -> Tuple[torch.Tensor]:
        sources, estimates = self.model(*self.example_inputs)
        sources = center_trim(sources, estimates)
        loss = self.criterion(estimates, sources)
        return (sources, estimates)

    def train(self):
        sources, estimates = self.model(*self.example_inputs)
        sources = center_trim(sources, estimates)
        loss = self.criterion(estimates, sources)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
