import pprint
import argparse
import gc

import torch
from torch.autograd import Variable

if __name__ == '__main__':
    from benchmark_common import benchmark_init
    from common import Bench
    from sru import SRU
else:
    from .benchmark_common import benchmark_init
    from .common import Bench
    from .sru import SRU


def run_sru(cpu=0, gpu=0, jit=False, use_kernel=False, backward=False,
            warmup=10, benchmark=20):
    assert not (jit and use_kernel)
    benchmark_init(0, 0, True)

    # input has length 20, batch size 32 and dimension 128
    x = Variable(torch.rand(20, 32, 128).cuda())
    input_size, hidden_size = 128, 128

    rnn = SRU(input_size, hidden_size,
              num_layers=2,          # number of stacking RNN layers
              dropout=0.00001,           # dropout applied between RNN layers
              rnn_dropout=0.0001,       # variational dropout applied on linear transformation
              use_tanh=1,            # use tanh?
              use_relu=0,            # use ReLU?
              bidirectional=False,    # bidirectional RNN ?
              use_kernel=use_kernel,
              jit=jit,
              )
    rnn.cuda()

    kernel_tag = '_kernel' if use_kernel else ''
    backward_tag = '_training' if backward else '_forward'
    jit_tag = '_jit' if jit else ''
    name = 'sru{}{}{}'.format(backward_tag, kernel_tag, jit_tag)
    iter_timer = Bench(cuda=True, name=name, warmup_iters=warmup)

    for _ in range(warmup + benchmark):
        gc.collect()
        with iter_timer:
            output, hidden = rnn(x)      # forward pass
            if backward:
                output.sum().backward()
        # output is (length, batch size, hidden size * number of directions)
        # hidden is (layers, batch size, hidden size * number of directions)
    return iter_timer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch mLSTM benchmark.")
    parser.add_argument('--cpu', type=int, default=0, help="CPU to run on")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on")
    parser.add_argument('--warmup', type=int, default=10, help="Warmup iterations")
    parser.add_argument('--benchmark', type=int, default=20, help="Benchmark iterations")
    parser.add_argument('--jit', action='store_true', help="Use JIT compiler")
    parser.add_argument('--use-kernel', action='store_true', help="Use specialized kernel")
    parser.add_argument('--backward', action='store_true', help="benchmark forward + backward")
    args = parser.parse_args()

    pprint.pprint(vars(args))
    run_sru(**vars(args))
