"""Run benchmark on ParlAI Memnn Model."""
import torch
from torch import nn
import gc
import argparse
import pprint


if __name__ == '__main__':
    from benchmark_common import benchmark_init
    from common import AttrDict, Bench, tag
    from models import memnn
else:
    from .benchmark_common import benchmark_init
    from .common import AttrDict, Bench, tag
    from .models import memnn


def one_to_many(query_embeddings, answer_embeddings, reply_embeddings=None):
    return query_embeddings.mm(answer_embeddings.t())


def run_memnn(warmup=2, benchmark=18, jit=False, cuda=False):
    nbatches = warmup + benchmark

    default_params = dict(lr=0.01, embedding_size=128, hops=3, mem_size=100,
                          time_features=False, position_encoding=True,
                          output='rank', dropout=0.1, optimizer='adam',
                          num_features=500, num_batches=nbatches, cuda=cuda)
    params = AttrDict(default_params)

    """Set up model."""
    # The CPU version is slow...
    params['batch_size'] = 4 if params.cuda else 4

    if params.cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = memnn.MemNN(params, params.num_features)
    criterion = nn.CrossEntropyLoss()
    data_batches = [
        [  # memories, queries, memory_lengths, query_lengths
            torch.zeros(params.batch_size * params.mem_size, dtype=torch.long, device=device),
            torch.zeros(params.batch_size * 28             , dtype=torch.long, device=device),
            torch.ones(params.batch_size, params.mem_size , dtype=torch.long, device=device),
            torch.full((params.batch_size,), 28            , dtype=torch.long, device=device),
        ]
        for _ in range(params.num_batches)
    ]
    cand_batches = [
        torch.zeros(params.batch_size * 14, params.embedding_size, device=device)
        for _ in range(params.num_batches)
    ]
    target_batches = [
        torch.ones(params.batch_size, dtype=torch.long, device=device)
        for _ in range(params.num_batches)
    ]

    # model.to(device) # embeddings are performed on CPU
    # the memnn model takes care of things when it is passed the cuda flag
    criterion.to(device)

    """Time model."""
    cuda_tag = '_cuda' if cuda else ''
    jit_tag = '_jit' if jit else ''
    name = 'memnn{}{}'.format(cuda_tag, jit_tag)
    bench = Bench(name=name, cuda=cuda, warmup_iters=warmup)
    trace_once = jit

    total_loss = 0
    for data, cands, targets in zip(data_batches, cand_batches, target_batches):
        gc.collect()
        if trace_once:
            model = torch.jit.trace(*data)(model)
            trace_once = False
        with bench:
            output_embeddings = model(*data)
            scores = one_to_many(output_embeddings, cands)
            loss = criterion(scores, targets)
            loss.backward()
            total_loss += float(loss.item())

    return bench


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch memnn bench")
    parser.add_argument('--warmup', type=int, default=2, help="Warmup iterations")
    parser.add_argument('--benchmark', type=int, default=10, help="Benchmark iterations")
    parser.add_argument('--jit', action='store_true', help="Use JIT compiler")
    parser.add_argument('--cuda', action='store_true', help="use cuda")
    args = parser.parse_args()
    pprint.pprint(vars(args))

    run_memnn(**vars(args))
