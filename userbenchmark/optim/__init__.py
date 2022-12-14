from typing import Any, Dict, List, Tuple
from torchbenchmark import load_model_by_name
import torch
from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, SGD, RAdam, Rprop, RMSprop, NAdam
import torch.utils.benchmark as benchmark
from userbenchmark.utils import dump_output, get_output_json
import argparse
import sys
import itertools


BM_NAME = 'optim'

MODEL_NAMES = [
    'BERT_pytorch',
    # 'hf_T5_large',
    'resnet18',
]

OPTIM_NAMES = [o.__name__ for o in [Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, SGD, RAdam, Rprop, RMSprop, NAdam]]

OPTIMIZERS = [
    # Adadelta(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0, foreach: Optional[bool] = None,
    #          maximize: bool = False, differentiable: bool = False)
    (Adadelta, {}),
    (Adadelta, {'maximize': True}),
    (Adadelta, {'foreach': True,}),
    (Adadelta, {'foreach': True, 'maximize': True}),
    # Adagrad(self, params, lr=1e-2, lr_decay=0, weight_decay=0, eps=1e-10, foreach: Optional[bool] = None, 
    #         maximize: bool = False, differentiable: bool = False)
    (Adagrad, {}),
    (Adagrad, {'maximize': True}),
    (Adagrad, {'foreach': True,}),
    (Adagrad, {'foreach': True, 'maximize': True}),
    # Adam(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    #      weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
    #      maximize: bool = False, capturable: bool = False,
    #      differentiable: bool = False, fused: bool = False):
    (Adam, {}),
    (Adam, {'amsgrad': True}),
    (Adam, {'maximize': True}),
    (Adam, {'foreach': True}),
    (Adam, {'foreach': True, 'maximize': True}),
    (Adam, {'foreach': True, 'amsgrad': True}),
    (Adam, {'foreach': True, 'capturable': True}),
    (Adam, {'fused': True}),
    (Adam, {'fused': True, 'amsgrad': True}),
    (Adam, {'fused': True, 'maximize': True}),
    (Adam, {'fused': True, 'capturable': True}),

    (AdamW, {}),
    (AdamW, {'maximize': True}),
    (AdamW, {'foreach': True}),
    (AdamW, {'foreach': True, 'maximize': True, 'capturable': True}),
    (Adamax, {}),
    (Adamax, {'maximize': True}),
    (Adamax, {'foreach': True,}),
    (Adamax, {'foreach': True, 'maximize': True}),
    (ASGD, {}),
    (ASGD, {'maximize': True}),
    (ASGD, {'foreach': True,}),
    (ASGD, {'foreach': True, 'maximize': True}),
    (SGD, {}),
    (SGD, {'maximize': True}),
    (SGD, {'foreach': True,}),
    (SGD, {'foreach': True, 'momentum': 0.9, 'nesterov': True}),
    (SGD, {'foreach': True, 'momentum': 0.9, }),
    (SGD, {'foreach': True, 'maximize': True}),
    (RAdam, {}),
    (RAdam, {'foreach': True,}),
    (Rprop, {}),
    (Rprop, {'maximize': True}),
    (Rprop, {'foreach': True,}),
    (Rprop, {'foreach': True, 'maximize': True}),
    (RMSprop, {}),
    (RMSprop, {'maximize': True}),
    (RMSprop, {'foreach': True,}),
    (RMSprop, {'foreach': True, 'maximize': True}),
    (NAdam, {}),
    (NAdam, {'foreach': True,}),

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

def defaults_to_str(defaults: Dict[str, Any]) -> str:
    def entry_to_str(k, v) -> str:
        if isinstance(v, bool):
            return '' if not v else k
        return f'{k}={v}'
    return ', '.join([entry_to_str(k, v) for k, v in defaults.items()])

def run_model(modelName, device, Optim, defaults):
    Model = load_model_by_name(modelName)   
    mod, inputs, params = get_model_deets(Model(device=device, test='train'))
    
    if Optim.__name__ == 'SGD':
        defaults['lr'] = 1e-2
    optim = Optim(params, **defaults)
    forward_and_backward(mod, inputs)

    return benchmark.Timer(
        stmt='optimizer_step(optim)',
        globals={'optim': optim, 'optimizer_step': optimizer_step},
        sub_label=f'{modelName} {optim.__class__.__name__}, {device}',
        description='default' if len(defaults) == 0 else defaults_to_str(defaults)
    ).blocked_autorange()

def run_benchmarks(optims: str) -> List[float]:
    print(optims)
    results = []
    for mn in MODEL_NAMES:
        for d in devices:
            for O, defaults in OPTIMIZERS:
                # fused/capturable requires params to be floats on CUDA
                if ('fused' in defaults and defaults['fused'] or 'capturable' in defaults and defaults['capturable']) and d != 'cuda':
                    continue
                if O.__name__ in optims:
                    results.append(run_model(mn, d, O, defaults))
    return results


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optim",
        nargs="*",
        default=OPTIM_NAMES,
        choices=OPTIM_NAMES,
        help="List of optimizers to run tests on")
    args = parser.parse_args(args)
    return args

def run(args: List[str]):
    args = parse_args(args)
    results = run_benchmarks(args.optim)
    metrics: Dict[str, float] = {} 
    # gotta output a JSON now do I
    dump_output(BM_NAME, get_output_json(BM_NAME, metrics))
    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize(rowwise=True)
    compare.print()

if __name__ == '__main__':
    run(sys.argv[1:])


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