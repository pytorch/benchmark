import torch
import torch.nn as nn
from torch.autograd import Variable
import time

from framework import Benchmark
from framework import utils


class CPULSTMBench(Benchmark):
    sizes = [
        [64, 15, 500, 500],
        [64, 20, 500, 500],
        [64, 25, 500, 500],
        [64, 30, 500, 500],
        [64, 35, 500, 500],
        [64, 40, 500, 500],
        [64, 45, 500, 500],
        [64, 50, 500, 500],
        [16, 25, 512, 512],
        [32, 25, 512, 512],
        [64, 25, 512, 512],
        [128, 25, 512, 512],
        [16, 25, 1024, 1024],
        [32, 25, 1024, 1024],
        [64, 25, 1024, 1024],
        [128, 25, 1024, 1024],
        [16, 25, 2048, 2048],
        [32, 25, 2048, 2048],
        [64, 25, 2048, 2048],
        [128, 25, 2048, 2048],
        [16, 25, 4096, 4096],
        [32, 25, 4096, 4096],
        [64, 25, 4096, 4096],
        [128, 25, 4096, 4096],
    ]
    args = utils.grid({"size": sizes, "train": (True, False)})
    user_counters = {
        "duration": 0,
        "gflops": 10 * " ",
        "GFLOPS": 10 * " ",
        "SPS": 10 * " ",
    }

    def setupRun(self, state, arg):
        size = arg.size

        N = size[0]  # batch size
        T = size[1]  # sentence length
        D = size[2]  # embedding size
        H = size[3]  # hidden size

        state.N, state.T, state.D, state.H = N, T, D, H

        state.rnn = nn.LSTM(D, H, 1)
        state.input = Variable(torch.randn(T, N, D))
        state.h0 = Variable(torch.randn(1, N, H))
        state.c0 = Variable(torch.randn(1, N, H))

        state.output, state.hn = state.rnn(state.input, (state.h0, state.c0))
        if arg.train:
            state.loss_fn = torch.nn.L1Loss()

        state.targets = Variable(torch.randn(T, N, D))
        state.num_iter = 0
        state.elapsed = 0

    def benchmark(self, state, arg):
        start = time.time()
        state.output, state.hn = state.rnn(state.input, (state.h0, state.c0))
        if arg.train:
            loss = state.loss_fn(state.output, state.targets)
            loss.backward()
        state.elapsed += time.time() - start
        state.num_iter += 1

    def teardownRun(self, state, arg):
        dura = (state.elapsed) / state.num_iter  # time of ONE iteration
        N, T, D, H = state.N, state.T, state.D, state.H
        gflops = T * 4 * (N * H * D * 2 + N * H * H * 2) / 1e9
        GFLOPS = gflops / dura  # giga floating-point operations per second
        SPS = N / dura  # number of processed sentences per second
        state.duration = "{:.4f}".format(dura)
        state.gflops = "{:.4f}".format(gflops)
        state.GFLOPS = "{:.4f}".format(GFLOPS)
        state.SPS = "{:.4f}".format(SPS)
