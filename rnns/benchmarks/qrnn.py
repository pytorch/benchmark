import argparse
import pprint
import gc
import torch


if __name__ == '__main__':
    from benchmark_common import benchmark_init
    from common import Bench, tag
    from torchqrnn import QRNN
else:
    from .benchmark_common import benchmark_init
    from .common import Bench, tag
    from .torchqrnn import QRNN


def run_qrnn(batch_size=20, input_size=128, seq_len=20,
             warmup=10, benchmark=10,
             hidden_size=256, num_layers=10,
             use_kernel=False, jit=False, cuda=False):
    assert not (use_kernel and jit)
    if use_kernel:
        assert cuda

    benchmark_init(0, 0, True)
    name = 'qrnn{}{}{}'.format(tag(cuda=cuda), tag(jit=jit),
                               tag(kernel=use_kernel))
    iter_timer = Bench(name=name, cuda=cuda, warmup_iters=warmup)
    niters = warmup + benchmark

    size = (seq_len, batch_size, input_size)
    if cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    batches = [torch.rand(size, requires_grad=True, device=device)
               for _ in range(niters)]
    qrnn = QRNN(input_size, hidden_size, num_layers=num_layers, dropout=0.4,
                use_kernel=use_kernel, jit=jit).to(device)

    for X in batches:
        gc.collect()
        with iter_timer:
            output, hidden = qrnn(X)
            output.sum().backward()

    return iter_timer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch qrnn benchmark.")
    parser.add_argument('--batch-size', type=int, default=20, help="Batch size")
    parser.add_argument('--input-size', type=int, default=128, help="Input size")
    parser.add_argument('--hidden-size', type=int, default=256, help="Hidden size")
    parser.add_argument('--num-layers', type=int, default=10, help="Hidden size")
    parser.add_argument('--seq-len', type=int, default=20, help="Sequence length")
    parser.add_argument('--warmup', type=int, default=10, help="Warmup iterations")
    parser.add_argument('--benchmark', type=int, default=20, help="Benchmark iterations")
    parser.add_argument('--cuda', action='store_true', help="Use cuda")
    parser.add_argument('--use-kernel', action='store_true', help="Use fused cell")
    parser.add_argument('--jit', action='store_true', help="Use JIT compiler")
    args = parser.parse_args()

    pprint.pprint(vars(args))
    run_qrnn(**vars(args))
