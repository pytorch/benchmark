import torch
from torch import nn
from torch.autograd import Variable
from . import *

import models.bnlstm as bnlstm

# From https://github.com/jihunchoi/recurrent-batch-normalization-pytorch


class BNLSTM(Benchmark):
    default_params = dict(hidden_size=100, max_length=784, pmnist=False, num_batches=1)
    params = make_params(cuda=over(True, False))

    def prepare(self, p):
        # The CPU version is slow...
        p['batch_size'] = 20 if p.cuda else 5
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.rnn = bnlstm.LSTM(cell_class=bnlstm.BNLSTMCell, input_size=1, hidden_size=p.hidden_size, batch_first=True, max_length=p.max_length)
                self.fc = nn.Linear(in_features=p.hidden_size, out_features=10) # 10 digits in mnist

            def forward(self, data):
                hx = None
                if not p.pmnist:
                    h0 = Variable(data.data.new(data.size(0), p.hidden_size)
                                  .normal_(0, 0.1))
                    c0 = Variable(data.data.new(data.size(0), p.hidden_size)
                                  .normal_(0, 0.1))
                    hx = (h0, c0)
                _, (h_n, _) = self.rnn(input_=data, hx = hx)
                logits = self.fc(h_n[0])
                return logits

        def cast(tensor):
            return tensor.cuda() if p.cuda else tensor

        self.model = Model()
        self.criterion = nn.CrossEntropyLoss()
        self.data_batches = [Variable(cast(torch.zeros(p.batch_size, 28 * 28, 1))) for _ in range(p.num_batches)]
        self.target_batches = [Variable(cast(torch.zeros(p.batch_size)).long()) for _ in range(p.num_batches)]
        if p.cuda:
            self.model.cuda()
            self.criterion.cuda()

    def time_bnlstm(self, p):
        total_loss = 0
        for data, targets in zip(self.data_batches, self.target_batches):
            logits = self.model(data)
            loss = self.criterion(input=logits, target=targets)
            loss.backward()
            total_loss += loss.data  # CUDA sync point
        if p.cuda:
            torch.cuda.synchronize()


