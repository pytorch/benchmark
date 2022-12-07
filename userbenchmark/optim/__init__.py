from typing import Any, Dict, List, Tuple
from torchbenchmark import load_model_by_name
import torch
import torch.utils.benchmark as benchmark
from userbenchmark.utils import dump_output, get_output_json


BM_NAME = 'optim'

MODEL_NAMES = [
    'BERT_pytorch',
    # 'hf_T5_large',
    'resnet18',
]

OPTIMIZERS = [
    (torch.optim.Adadelta, {'foreach': True,}),
    (torch.optim.Adagrad, {'foreach': True,}),
    (torch.optim.Adam, {'foreach': True, 'fused': True}),
    (torch.optim.AdamW, {'foreach': True,}),
    (torch.optim.Adamax, {'foreach': True,}),
    (torch.optim.ASGD, {'foreach': True,}),
    (torch.optim.SGD, {'foreach': True,}),
    (torch.optim.RAdam, {'foreach': True,}),
    (torch.optim.Rprop, {'foreach': True,}),
    (torch.optim.RMSprop, {'foreach': True,}),
    (torch.optim.NAdam, {'foreach': True,}),

    ## don't run the below, as they don't work
    # (torch.optim.SparseAdam, {}),
    # (torch.optim.LBFGS, {}),
]

devices = ['cuda', 'cpu']

def get_model_deets(m) -> Tuple[Any, Any, Any]: 
    model, inputs = m.get_module()
    return model, inputs, model.parameters()

def forward_and_backward(mod, inputs):
    pred = mod(*inputs)
    loss = torch.sum(pred)
    loss.backward()

def optimizer_step(optimizer):
    optimizer.step()

def run_model(modelName, device, Optim, foreach, fused):
    Model = load_model_by_name(modelName)   
    mod, inputs, params = get_model_deets(Model(device=device, test='train'))
    defaults = {}
    if Optim.__name__ == 'SGD':
        defaults['lr'] = 1e-2
    if foreach:
        defaults['foreach'] = True
    if fused:
        defaults['fused'] = True
    optim = Optim(params, **defaults)
    forward_and_backward(mod, inputs)
    # optimizer_step(optim)

    return benchmark.Timer(
        stmt='optimizer_step(optim)',
        globals={'optim': optim, 'optimizer_step': optimizer_step},
        sub_label=f'{modelName} {optim.__class__.__name__}, {device}',
        description='fused' if fused else ('foreach' if foreach else 'default')
    ).blocked_autorange()

def run_benchmarks() -> List[float]:
    results = []
    for mn in MODEL_NAMES:
        for d in devices:
            for O, impls in OPTIMIZERS:
                results.append(run_model(mn, d, O, None, None))
                if 'foreach' in impls and impls['foreach']:
                    results.append(run_model(mn, d, O, True, None))
                # fused requires params to be floats on CUDA
                if 'fused' in impls and impls['fused'] and d == 'cuda':
                    results.append(run_model(mn, d, O, None, True))
    return results


def run(args: List[str]):
    results = run_benchmarks()
    metrics: Dict[str, float] = {} 
    # gotta output a JSON now do I
    dump_output(BM_NAME, get_output_json(BM_NAME, metrics))
    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize(rowwise=True)
    compare.print()

if __name__ == '__main__':
    run([])


# Steps to longterm benchmarks:
# 1. add a knob for users to provide their own optimizer (set_optimizer API) => 1 day or so
# 2. if we want to just benchmark the optimizer part, we need to split the training loop into forward/backward/optim
#     => ~20 models would be done within a week, others can run into exceptions since some models don't follow the same structure

# OR do the torchdynamo approach


# Difference between pytorch/pytorch benchmarks/dynamo and pytorch/torchbench/userbenchmark
# Ownership. If you wanna leverage torchbench tools, userbenchmark is better, esp to set up 
# nightly CI/internal dashboards. 
# You're on your own for outside the repo benchmarks.

# How do I get parameters from these models
# use get_module to 