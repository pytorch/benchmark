import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time

from framework import GridBenchmark, AttrDict

# TODO: Setup mobilenet
# from misc.mobilenet import MobileNetV2
# models.__dict__["mobilenet_v2"] = MobileNetV2


class Convnets(GridBenchmark):
    args = {
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
        "inference": (True, True),
        "cuda": (False,),
    }
    user_counters = {
        "time_fwd_avg": 0,
        "time_bwd_avg": 0,
        "time_upt_avg": 0,
        "time_total": 0,
    }

    def setup(self, state, arg):
        if arg.cuda:
            import torch.backends.cudnn as cudnn

            cudnn.benchmark = True
        state.counters = AttrDict()

    def benchmark(self, state, arg):
        steps = 10  # nb of steps in loop to average perf
        nDryRuns = 5
        arch, sizes = arg[("arch", "size")]
        batch_size, c, h, w = sizes[0], sizes[1], sizes[2], sizes[3]
        batch_size = 1 if arg.single_batch_size else batch_size

        data_ = torch.randn(batch_size, c, h, w)
        target_ = torch.arange(1, batch_size + 1).long()
        net = models.__dict__[
            arch
        ]()  # no need to load pre-trained weights for dummy data

        optimizer = optim.SGD(net.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        if arg.cuda:
            data_, target_ = data_.cuda(), target_.cuda()
            net.cuda()
            criterion = criterion.cuda()

        net.eval()

        # print(
        #     "ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%d"
        #     % (arch, kernel, batch_size, c, h, w)
        # )

        data, target = Variable(data_), Variable(target_)

        for i in range(nDryRuns):
            optimizer.zero_grad()  # zero the gradient buffers
            output = net(data)
            if not arg.inference:
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()  # Does the update

        time_fwd, time_bwd, time_upt = 0, 0, 0

        for i in range(steps):
            optimizer.zero_grad()  # zero the gradient buffers
            t1 = time.time()
            output = net(data)
            t2 = time.time()
            if not arg.inference:
                loss = criterion(output, target)
                loss.backward()
                t3 = time.time()
                optimizer.step()  # Does the update
                t4 = time.time()
            time_fwd = time_fwd + (t2 - t1)
            if not arg.inference:
                time_bwd = time_bwd + (t3 - t2)
                time_upt = time_upt + (t4 - t3)

        time_fwd_avg = time_fwd / steps * 1000
        time_bwd_avg = time_bwd / steps * 1000
        time_upt_avg = time_upt / steps * 1000

        # update not included!
        time_total = time_fwd_avg + time_bwd_avg

        state.time_fwd_avg = "{:2.3f}".format(time_fwd_avg)
        state.time_bwd_avg = "{:2.3f}".format(time_bwd_avg)
        state.time_upt_avg = "{:2.3f}".format(time_upt_avg)
        state.time_total = "{:2.3f}".format(time_total)
        # print(
        #     "%-30s %10s %10.2f %10.2f"
        #     % (
        #         kernel,
        #         ":forward:",
        #         time_fwd_avg,
        #         batch_size * 1000 / time_fwd_avg,
        #     )
        # )
        # print("%-30s %10s %10.2f" % (kernel, ":backward:", time_bwd_avg))
        # print("%-30s %10s %10.2f" % (kernel, ":update:", time_upt_avg))
        # print(
        #     "%-30s %10s %10.2f %10.2f"
        #     % (kernel, ":total:", time_total, batch_size * 1000 / time_total)
        # )
