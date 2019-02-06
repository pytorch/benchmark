import torch
from torch.autograd import Variable
import torch.jit

import argparse
import pprint
import gc
import time

if __name__ == '__main__':
    from benchmark_common import benchmark_init
    from common import Bench, tag
else:
    from .benchmark_common import benchmark_init
    from .common import Bench, tag


def mlstm_raw(input, hx, cx, w_xm, w_hm, w_ih, w_mh):
    # w_ih holds W_hx, W_ix, W_ox, W_fx
    # w_mh holds W_hm, W_im, W_om, W_fm

    m = input.mm(w_xm.t()) * hx.mm(w_hm.t())
    gates = input.mm(w_ih.t()) + m.mm(w_mh.t())

    ingate, forgetgate, hiddengate, outgate = gates.chunk(4, 1)

    ingate = ingate.sigmoid()
    outgate = outgate.sigmoid()
    forgetgate = forgetgate.sigmoid()

    cy = (forgetgate * cx) + (ingate * hiddengate)
    hy = (cy * outgate).tanh()

    return hy, cy


def run_mlstm(cpu=0, gpu=0, batch_size=1, input_size=205, hidden_size=1900, embed_size=None,
              seq_len=20, warmup=10, benchmark=20, autograd=False, jit=False,
              backward=False, skip_cpu_governor_check=False):
    name = "mlstm_jit" if jit else "mlstm"
    iter_timer = Bench(name=name, cuda=True, warmup_iters=warmup)

    if embed_size is None:
        embed_size = hidden_size

    if jit or backward:
        autograd = True

    benchmark_init(cpu, gpu, skip_cpu_governor_check)

    requires_grad = autograd
    device = torch.device(gpu)

    input = torch.randn(seq_len, batch_size, input_size, requires_grad=requires_grad, device=device)
    hx = torch.randn(batch_size, hidden_size, requires_grad=requires_grad, device=device)
    cx = torch.randn(batch_size, hidden_size, requires_grad=requires_grad, device=device)
    w_xm = torch.randn(embed_size, input_size, requires_grad=requires_grad, device=device)
    w_hm = torch.randn(embed_size, hidden_size, requires_grad=requires_grad, device=device)
    w_ih = torch.randn(4 * hidden_size, input_size, requires_grad=requires_grad, device=device)
    w_mh = torch.randn(4 * hidden_size, embed_size, requires_grad=requires_grad, device=device)
    params = [input, hx, cx, w_xm, w_hm, w_ih, w_mh]

    if jit:
        mlstm = torch.jit.trace(input[0], hx, cx, w_xm, w_hm, w_ih, w_mh)(mlstm_raw)
    else:
        mlstm = mlstm_raw

    for _ in range(warmup + benchmark):
        gc.collect()
        with iter_timer:
            hx_t = hx
            cx_t = cx
            for j in range(seq_len):
                hx_t, cx_t = mlstm(input[j], hx_t, cx_t, w_xm, w_hm, w_ih, w_mh)
            if backward:
                hx_t.sum().backward()
                for param in params:
                    param.grad.zero_()

    return iter_timer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch mLSTM benchmark.")
    parser.add_argument('--cpu', type=int, default=0, help="CPU to run on")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size")
    parser.add_argument('--input-size', type=int, default=205, help="Input size")
    parser.add_argument('--hidden-size', type=int, default=1900, help="Hidden size")
    parser.add_argument('--embed-size', type=int, default=None, help="Embed size")
    parser.add_argument('--seq-len', type=int, default=20, help="Sequence length")
    parser.add_argument('--warmup', type=int, default=10, help="Warmup iterations")
    parser.add_argument('--benchmark', type=int, default=20, help="Benchmark iterations")
    parser.add_argument('--autograd', action='store_true', help="Use autograd")
    parser.add_argument('--jit', action='store_true', help="Use JIT compiler (implies --autograd)")
    parser.add_argument('--backward', action='store_true', help="benchmark forward + backward (implies --autograd)")
    parser.add_argument('--skip-cpu-governor-check', action='store_true',
                        help="Skip checking whether CPU governor is set to `performance`")
    args = parser.parse_args()

    pprint.pprint(vars(args))
    run_mlstm(**vars(args))
