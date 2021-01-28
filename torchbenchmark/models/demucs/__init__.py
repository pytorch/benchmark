import json
import torch
import random
import numpy as np
from fractions import Fraction
from torchbenchmark.tasks import OTHER

from .demucs.model import Demucs
from .demucs.parser import get_name, get_parser
from .demucs.augment import FlipChannels, FlipSign, Remix, Shift
from .demucs.utils import capture_init, center_trim


torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Model:
    task = OTHER.OTHER_TASKS
    def __init__(self, device=None, jit=False):
        self.device = device
        self.jit = jit
        self.parser = get_parser()
        self.args = self.parser.parse_args([])
        args = self.args
        self.model = Demucs(channels=32)  # Change the channel to 32 to fit 16-GB GPU
        self.dmodel = self.model
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        if 1:
            samples = 80000
            # TODO: calculate the right shape
            self.example_inputs = (torch.rand([4, 5, 2, 135576]), )

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

    def get_module(self):
        # TODO: merge this with train and eval
        def helper(streams):
            streams = streams.to(self.device)
            sources = streams[:, 1:]
            sources = self.augment(sources)
            mix = sources.sum(dim=1)
            return self.model(mix)
        return helper, self.example_inputs

    def eval(self, niter=1):
        # TODO: implement the eval version
        self.model.eval()
        for _ in range(niter):
            streams, = self.example_inputs
            streams = streams.to(self.device)
            sources = streams[:, 1:]
            sources = self.augment(sources)
            mix = sources.sum(dim=1)

            estimates = self.model(mix)
            sources = center_trim(sources, estimates)
            loss = self.criterion(estimates, sources)

    def train(self, niter=1):
        self.model.train()
        for _ in range(niter):
            streams, = self.example_inputs
            streams = streams.to(self.device)
            sources = streams[:, 1:]
            sources = self.augment(sources)
            mix = sources.sum(dim=1)

            estimates = self.model(mix)
            sources = center_trim(sources, estimates)
            loss = self.criterion(estimates, sources)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    m.train(niter=1)
    m.eval(niter=1)
