import torch
from torch import nn
from torch.autograd import Variable
import argparse
import pprint
import gc

if __name__ == '__main__':
    from benchmark_common import benchmark_init
    from common import Bench, tag
    from models import bnlstm
else:
    from .benchmark_common import benchmark_init
    from .common import Bench, tag
    from .models import bnlstm


# From https://github.com/jihunchoi/recurrent-batch-normalization-pytorch

def run_bnlstm(hidden_size=100, max_length=784, pmnist=False, num_batches=5,
               cuda=False, jit=False, warmup=10, benchmark=20):
    name = 'bnlstm{}{}'.format(tag(cuda=cuda), tag(jit=jit))
    iter_timer = Bench(name, cuda=cuda, warmup_iters=2)

    # The CPU version is slow...
    batch_size = 20 if cuda else 5

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.rnn = bnlstm.LSTM(cell_class=bnlstm.BNLSTMCell, input_size=1,
                                   hidden_size=hidden_size, batch_first=True,
                                   max_length=max_length, jit=jit)
            self.fc = nn.Linear(in_features=hidden_size, out_features=10)  # 10 digits in mnist

        def forward(self, data):
            hx = None
            if not pmnist:
                h0 = Variable(data.data.new(data.size(0), hidden_size)
                              .normal_(0, 0.1))
                c0 = Variable(data.data.new(data.size(0), hidden_size)
                              .normal_(0, 0.1))
                hx = (h0, c0)
            _, (h_n, _) = self.rnn(input_=data, hx=hx)
            logits = self.fc(h_n[0])
            return logits

    def cast(tensor):
        return tensor.cuda() if cuda else tensor

    model = Model()
    criterion = nn.CrossEntropyLoss()
    data_batches = [Variable(cast(torch.zeros(batch_size, 28 * 28, 1))) for _ in range(num_batches)]
    target_batches = [Variable(cast(torch.zeros(batch_size)).long()) for _ in range(num_batches)]
    if cuda:
        model.cuda()
        criterion.cuda()

    total_loss = 0
    for data, targets in zip(data_batches, target_batches):
        gc.collect()
        with iter_timer:
            logits = model(data)
            loss = criterion(input=logits, target=targets)
            loss.backward()
            total_loss += float(loss.data.item())  # CUDA sync point

    return iter_timer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch BNLSTM benchmark.")
    parser.add_argument('--num_batches', type=int, default=1, help="num batches")
    parser.add_argument('--hidden-size', type=int, default=100, help="Hidden size")
    parser.add_argument('--max-length', type=int, default=784, help="max seq len")
    parser.add_argument('--warmup', type=int, default=10, help="Warmup iterations")
    parser.add_argument('--benchmark', type=int, default=20, help="Benchmark iterations")
    parser.add_argument('--jit', action='store_true', help="Use JIT")
    parser.add_argument('--cuda', action='store_true', help="Use cuda")
    args = parser.parse_args()

    pprint.pprint(vars(args))
    run_bnlstm(**vars(args))
