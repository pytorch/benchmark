import torch as th
from torch.autograd import Variable as V
import gc

import benchmarks.lstm_variants as lstm_variants

if __name__ == '__main__':
    from benchmark_common import benchmark_init
    from common import AttrDict, Bench, tag
else:
    from .benchmark_common import benchmark_init
    from .common import AttrDict, Bench, tag


lstms = [
    'SlowLSTM',
    'LSTM',
    'GalLSTM',
    'MoonLSTM',
    'SemeniutaLSTM',
    'LayerNormLSTM',
    'LayerNormGalLSTM',
    'LayerNormMoonLSTM',
    'LayerNormSemeniutaLSTM',
]

BATCH = 10
SEQ_LEN = 100
DROPOUT = 0.5


def run_lstm_variant(variant='SlowLSTM', cuda=False, size=128, jit=False):
    assert variant in lstms
    p = AttrDict({'cuda': cuda, 'lstm_kind': variant, 'size': size})

    name = '{}_size{}{}{}'.format(variant, size, tag(cuda=cuda), tag(jit=jit))

    def C(x):
        if p.cuda:
            x = x.cuda()
        return x

    lstm = getattr(lstm_variants, p.lstm_kind)
    x = V(C(th.rand(1, BATCH, p.size)))
    hiddens = (V(C(th.rand(1, BATCH, p.size))), V(C(th.rand(1, BATCH, p.size))))
    th.manual_seed(1234)
    cus = C(lstm(p.size, p.size, dropout=DROPOUT, jit=jit))
    if hasattr(cus, 'mask'):
        cus.mask = C(cus.mask)

    iter_timer = Bench(name=name, cuda=cuda, warmup_iters=3)

    # Super slow on CPU
    iters = 20 if cuda else 6
    for _ in range(iters):
        gc.collect()
        with iter_timer:
            out, h = x, hiddens
            for i in range(SEQ_LEN):
                out, h = cus(out, h)

    return iter_timer
