"""Run benchmark on ParlAI Memnn Model."""
import torch
from torch import nn
import gc
from .common import AttrDict, Bench
from .models import memnn
import argparse


def one_to_many(query_embeddings, answer_embeddings, reply_embeddings=None):
    return query_embeddings.mm(answer_embeddings.t())


def run_memnn(args):
    nbatches = args.warmup + args.benchmark

    default_params = dict(lr=0.01, embedding_size=128, hops=3, mem_size=100,
                          time_features=False, position_encoding=True,
                          output='rank', dropout=0.1, optimizer='adam',
                          num_features=500, num_batches=nbatches, cuda=False)
    params = AttrDict(default_params)

    """Set up model."""
    # The CPU version is slow...
    params['batch_size'] = 32 if params.cuda else 4

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
            torch.ones (params.batch_size, params.mem_size , dtype=torch.long, device=device),
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
    if params.cuda:
        model.cuda()
        criterion.cuda()

    """Time model."""
    bench = Bench(name='memnn')
    trace_once = args.jit

    total_loss = 0
    for data, cands, targets in zip(data_batches, cand_batches, target_batches):
        gc.collect()
        bench.start_timing()
        if trace_once:
            model = torch.jit.trace(*data)(model)
            trace_once = False
        output_embeddings = model(*data)
        scores = one_to_many(output_embeddings, cands)
        loss = criterion(scores, targets)
        loss.backward()
        total_loss += float(loss.item())
        bench.stop_timing()

    return bench


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch memnn bench")
    parser.add_argument('--warmup',     type=int, default=2,   help="Warmup iterations")
    parser.add_argument('--benchmark',  type=int, default=10,  help="Benchmark iterations")
    parser.add_argument('--jit',        action='store_true',   help="Use JIT compiler")
    args = parser.parse_args()

    run_memnn(args)
