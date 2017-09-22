import torch
from torch import nn
from torch.autograd import Variable
from . import *


class WLM(Benchmark):
    default_params = dict(rnn_type='LSTM', num_tokens=10000, embedding_size=200,
                          hidden_size=200, num_layers=2, batch_size=20, bptt=35,
                          dropout=0.5, num_batches=10, cuda=True)
    params = make_params(cuda=over(True, False))

    def prepare(self, rnn_type, num_tokens, embedding_size, hidden_size,
                num_layers, batch_size, bptt, dropout, num_batches, cuda):
        def get_rnn():
            if rnn_type in ['LSTM', 'GRU']:
                return getattr(nn, rnn_type)(embedding_size, hidden_size, num_layers, dropout=dropout)
            else:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                return nn.RNN(embedding_size, hidden_size, num_layers, nonlinearity=nonlinearity, dropout=dropout)

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.drop = nn.Dropout(dropout)
                self.rnn = get_rnn()
                self.encoder = nn.Embedding(num_tokens, embedding_size)
                self.decoder = nn.Linear(hidden_size, num_tokens)

            def forward(self, input):
                emb = self.drop(self.encoder(input))
                output, hidden = self.rnn(emb)
                output = self.drop(output)
                decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
                return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

        def cast(tensor):
            return tensor.long().cuda() if cuda else tensor.long()

        self.cuda = cuda
        self.model = Model()
        self.criterion = nn.CrossEntropyLoss()
        self.data_batches = [Variable(cast(torch.zeros(bptt, batch_size))) for _ in range(num_batches)]
        self.target_batches = [Variable(cast(torch.zeros(bptt * batch_size))) for _ in range(num_batches)]
        if cuda:
            self.model.cuda()
            self.criterion.cuda()

    def time_word_language_model_example(self, *args, **kwargs):
        total_loss = 0
        for data, targets in zip(self.data_batches, self.target_batches):
            output, _ = self.model(data)
            loss = self.criterion(output.view(-1, output.size(2)), targets)
            loss.backward()
            total_loss += loss.data  # CUDA sync point
        if self.cuda:
            torch.cuda.synchronize()
