"""Run benchmark on ParlAI Memnn Model."""
import torch
from torch import nn
from torch.autograd import Variable
from . import Benchmark, make_params, over, AttrDict
from models import memnn


class Memnn(Benchmark):
    """Memnn benchmark."""
    default_params = dict(lr=0.01, embedding_size=128, hops=3, mem_size=100,
                          time_features=False, position_encoding=True,
                          output='rank', dropout=0.1, optimizer='adam',
                          num_features=500, num_batches=1)
    params = make_params(cuda=over(True, False))

    def prepare(self, p):
        """Set up model."""
        # The CPU version is slow...
        p['batch_size'] = 32 if p.cuda else 4

        def cast(tensor):
            return tensor.cuda() if p.cuda else tensor

        self.model = memnn.MemNN(p, p.num_features)
        self.criterion = nn.CrossEntropyLoss()
        self.data_batches = [
            [  # memories, queries, memory_lengths, query_lengths
                Variable(cast(torch.zeros(p.batch_size * p.mem_size).long())),
                Variable(cast(torch.zeros(p.batch_size * 28).long())),
                Variable(cast(torch.ones(p.batch_size, p.mem_size).long())),
                Variable(cast(torch.LongTensor(p.batch_size).fill_(28).long())),
            ]
            for _ in range(p.num_batches)
        ]
        self.cand_batches = [
            Variable(cast(torch.zeros(p.batch_size * 14, p.embedding_size)))
            for _ in range(p.num_batches)
        ]
        self.target_batches = [
            Variable(cast(torch.ones(p.batch_size).long()))
            for _ in range(p.num_batches)
        ]
        if p.cuda:
            self.model.cuda()
            self.criterion.cuda()

    def time_memnn(self, p):
        """Time model."""
        total_loss = 0
        for data, cands, targets in zip(self.data_batches, self.cand_batches, self.target_batches):
            output_embeddings = self.model(*data)
            scores = self.model.score.one_to_many(output_embeddings, cands)
            loss = self.criterion(scores, targets)
            loss.backward()
            total_loss += loss.data
            if p.cuda:
                torch.cuda.synchronize()

if __name__ == '__main__':
    d = Memnn.default_params.copy()
    d['cuda'] = False
    p = AttrDict(d)
    m = Memnn()
    m.prepare(p)
    m.time_memnn(p)
