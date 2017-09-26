from . import *

import torch
from torch.autograd import Variable
from sru import SRU, SRUCell

class SRUTest(Benchmark):
    default_params = dict()
    params = make_params(use_kernel=over(False, True))

    def prepare(self, p):
        # input has length 20, batch size 32 and dimension 128
        self.x = Variable(torch.rand(20, 32, 128).cuda())
        input_size, hidden_size = 128, 128

        self.rnn = SRU(input_size, hidden_size,
            num_layers = 2,          # number of stacking RNN layers
            dropout = 0.00001,           # dropout applied between RNN layers
            rnn_dropout = 0.0001,       # variational dropout applied on linear transformation
            use_tanh = 1,            # use tanh?
            use_relu = 0,            # use ReLU?
            bidirectional = False,    # bidirectional RNN ?
            use_kernel=p.use_kernel,
        )
        self.rnn.cuda()

    def time_sru(self, p):
        output, hidden = self.rnn(self.x)      # forward pass
        # output is (length, batch size, hidden size * number of directions)
        # hidden is (layers, batch size, hidden size * number of directions)
