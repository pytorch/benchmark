from . import *
import torch as th
import torch.nn as nn
from torch.autograd import Variable as V

from time import time

import lstm_variants

lstms = over(
    'SlowLSTM',
    'LSTM',
    'GalLSTM',
    'MoonLSTM',
    'SemeniutaLSTM',
    'LayerNormLSTM',
    'LayerNormGalLSTM',
    'LayerNormMoonLSTM',
    'LayerNormSemeniutaLSTM',
)

N_ITER = 100
DROPOUT = 0.5



class LSTMVariants(Benchmark):
    default_params = dict()
    params = make_params(cuda=over(False, True), lstm_kind=lstms, size=over(128, 512))

    def prepare(self, p):
        def C(x):
            if p.cuda:
                x = x.cuda()
            return x
        lstm = getattr(lstm_variants,p.lstm_kind)
        self.x = V(C(th.rand(1, 1, p.size)))
        self.hiddens = (V(C(th.rand(1, 1, p.size))), V(C(th.rand(1, 1, p.size))))
        th.manual_seed(1234)
        self.cus = C(lstm(p.size, p.size, dropout=DROPOUT))
        if hasattr(self.cus, 'mask'):
            self.cus.mask = C(self.cus.mask)

    def time_lstm_variants(self, p):
        out, h = self.x, self.hiddens
        for i in range(N_ITER):
            out, h = self.cus(out, h)
