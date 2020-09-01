import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time

from framework import Benchmark
from framework import utils

# TODO: Setup mobilenet
# from misc.mobilenet import MobileNetV2
# models.__dict__["mobilenet_v2"] = MobileNetV2


class CPUConvnets(Benchmark):
    args = utils.grid({
        ("arch", "size"): (
            ("alexnet", (128, 3, 224, 224)),
            ("vgg11", (64, 3, 224, 224)),
            ("inception_v3", (128, 3, 299, 299)),
            ("resnet50", (128, 3, 224, 224)),
            ("squeezenet1_0", (128, 3, 224, 224)),
            ("densenet121", (32, 3, 224, 224)),
            # ("mobilenet_v2", (128, 3, 224, 224)),
        ),
        "single_batch_size": (True, False),
        "inference": (True, False),
    })
    user_counters = {
        "time_fwd_avg": 0,
        "time_bwd_avg": 0,
        "time_upt_avg": 0,
        "time_total": 0,
    }

    def setupRun(self, state, arg):
        arch, sizes = arg[("arch", "size")]
        batch_size, c, h, w = sizes[0], sizes[1], sizes[2], sizes[3]
        batch_size = 1 if arg.single_batch_size else batch_size

        data_ = torch.randn(batch_size, c, h, w)
        target_ = torch.arange(1, batch_size + 1).long()
        state.net = models.__dict__[
            arch
        ]()  # no need to load pre-trained weights for dummy data

        state.optimizer = optim.SGD(state.net.parameters(), lr=0.01)
        state.criterion = nn.CrossEntropyLoss()

        state.net.eval()

        state.data, state.target = Variable(data_), Variable(target_)

        state.steps = 0
        state.time_fwd = 0
        state.time_bwd = 0
        state.time_upt = 0

    def benchmark(self, state, arg):
        state.optimizer.zero_grad()  # zero the gradient buffers
        t1 = time.time()
        output = state.net(state.data)
        t2 = time.time()
        if not arg.inference:
            loss = state.criterion(output, state.target)
            loss.backward()
            t3 = time.time()
            state.optimizer.step()  # Does the update
            t4 = time.time()
        state.time_fwd += t2 - t1
        if not arg.inference:
            state.time_bwd += t3 - t2
            state.time_upt += t4 - t3
        state.steps += 1

    def teardownRun(self, state, arg):

        time_fwd_avg = state.time_fwd / state.steps * 1000
        time_bwd_avg = state.time_bwd / state.steps * 1000
        time_upt_avg = state.time_upt / state.steps * 1000

        # update not included!
        time_total = time_fwd_avg + time_bwd_avg

        state.time_fwd_avg = "{:2.3f}".format(time_fwd_avg)
        state.time_bwd_avg = "{:2.3f}".format(time_bwd_avg)
        state.time_upt_avg = "{:2.3f}".format(time_upt_avg)
        state.time_total = "{:2.3f}".format(time_total)
