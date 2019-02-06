import torch
from torch import nn
from torch.autograd import Variable
from . import *

# This file is not in use


class WLM(Benchmark):
    default_params = dict(rnn_type='LSTM', num_tokens=10000, embedding_size=200,
                          hidden_size=200, num_layers=2, batch_size=20, bptt=35,
                          dropout=0.5, num_batches=10, cuda=True)
    params = make_params(cuda=over(True, False))

    def prepare(self, p):
        def get_rnn():
            if p.rnn_type in ['LSTM', 'GRU']:
                return getattr(nn, p.rnn_type)(p.embedding_size, p.hidden_size, p.num_layers, dropout=p.dropout)
            else:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[p.rnn_type]
                return nn.RNN(p.embedding_size, p.hidden_size, p.num_layers, nonlinearity=nonlinearity,
                              dropout=p.dropout)

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.drop = nn.Dropout(p.dropout)
                self.rnn = get_rnn()
                self.encoder = nn.Embedding(p.num_tokens, p.embedding_size)
                self.decoder = nn.Linear(p.hidden_size, p.num_tokens)

            def forward(self, input):
                emb = self.drop(self.encoder(input))
                output, hidden = self.rnn(emb)
                output = self.drop(output)
                decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
                return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

        def cast(tensor):
            return tensor.long().cuda() if p.cuda else tensor.long()

        self.model = Model()
        self.criterion = nn.CrossEntropyLoss()
        self.data_batches = [Variable(cast(torch.zeros(p.bptt, p.batch_size))) for _ in range(p.num_batches)]
        self.target_batches = [Variable(cast(torch.zeros(p.bptt * p.batch_size))) for _ in range(p.num_batches)]
        if p.cuda:
            self.model.cuda()
            self.criterion.cuda()

    def time_word_language_model_example(self, p):
        total_loss = 0
        for data, targets in zip(self.data_batches, self.target_batches):
            output, _ = self.model(data)
            loss = self.criterion(output.view(-1, output.size(2)), targets)
            loss.backward()
            total_loss += loss.data  # CUDA sync point
        if p.cuda:
            torch.cuda.synchronize()
