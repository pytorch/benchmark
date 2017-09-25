import torch
from torch.autograd import Variable
from sru import SRU, SRUCell

class SRUTest():

    def prepare(self, p):
        # input has length 20, batch size 32 and dimension 128
        self.x = Variable(torch.rand(20, 32, 128).cuda())
        input_size, hidden_size = 128, 128

        self.rnn = SRU(input_size, hidden_size,
            num_layers = 2,          # number of stacking RNN layers
            dropout = 0.0,           # dropout applied between RNN layers
            rnn_dropout = 0.0,       # variational dropout applied on linear transformation
            use_tanh = 1,            # use tanh?
            use_relu = 0,            # use ReLU?
            bidirectional = False    # bidirectional RNN ?
        )
        self.rnn.cuda()

    def test_sru(self, p):
        output, hidden = self.rnn(self.x)      # forward pass
        print(output,hidden)
        # output is (length, batch size, hidden size * number of directions)
        # hidden is (layers, batch size, hidden size * number of directions)
