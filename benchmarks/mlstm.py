import torch
from torch import nn
from torch.autograd import Variable
from . import *

import models.mlstm as mlstm

# From https://discuss.pytorch.org/t/implementation-of-multiplicative-lstm/2328/5

N_ITER = 700

class MultiplicativeLSTM(Benchmark):
    # parameters taken from the paper
    default_params = dict(batch_size=3, input_size=100, hidden_size=400, embed_size=400, cuda=True)
    params = make_params(cuda=over(True, False))

    def prepare(self, p):
        def cast(tensor):
            return tensor.cuda() if p.cuda else tensor

        self.input = Variable(cast(torch.randn(p.batch_size, p.input_size)))
        self.hiddens = (Variable(cast(torch.randn(p.batch_size, p.hidden_size))),
                        Variable(cast(torch.randn(p.batch_size, p.hidden_size))))
        self.w_xm = Variable(cast(torch.randn(p.embed_size, p.input_size)))
        self.w_hm = Variable(cast(torch.randn(p.embed_size, p.hidden_size)))
        self.w_ih = Variable(cast(torch.randn(4 * p.hidden_size, p.input_size)))
        self.w_mh = Variable(cast(torch.randn(4 * p.hidden_size, p.embed_size)))

    def time_mlstm(self, p):
        # TODO: this is totally bogus
        h = self.hiddens
        for i in range(N_ITER):
            # TODO: Don't keep using the same input
            h = mlstm.MultiplicativeLSTMCell(self.input, h, self.w_xm, self.w_hm, self.w_ih, self.w_mh)
