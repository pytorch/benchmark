from typing import Any, Dict, List, Tuple
from torchbenchmark import load_model_by_name
import torch
from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, SGD, RAdam, Rprop, RMSprop, NAdam, SparseAdam, LBFGS
import torch._dynamo as torchdynamo
import torch.utils.benchmark as benchmark
from userbenchmark.utils import dump_output, get_output_json
import argparse
import sys
import itertools


BM_NAME = 'optim'

MODEL_NAMES = [
    'BERT_pytorch',
    'hf_T5_large',
    'resnet18',
]

OPTIM_NAMES = [o.__name__ for o in [Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, SGD, RAdam, Rprop, RMSprop, NAdam, SparseAdam]]

FUNC_STRS = ['pt2_' , '']

OPTIMIZERS = [
    # Adadelta(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0, foreach: Optional[bool] = None,
    #          maximize: bool = False, differentiable: bool = False)
    (Adadelta, {}),
    # (Adadelta, {'maximize': True}),
    (Adadelta, {'foreach': False,}),
    (Adadelta, {'foreach': True,}),
    # (Adadelta, {'foreach': True, 'maximize': True}),
    # Adagrad(self, params, lr=1e-2, lr_decay=0, weight_decay=0, eps=1e-10, foreach: Optional[bool] = None, 
    #         maximize: bool = False, differentiable: bool = False)
    (Adagrad, {}),
    # (Adagrad, {'maximize': True}),
    (Adagrad, {'foreach': False,}),
    (Adagrad, {'foreach': True,}),
    # (Adagrad, {'foreach': True, 'maximize': True}),
    # Adam(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    #      weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
    #      maximize: bool = False, capturable: bool = False,
    #      differentiable: bool = False, fused: bool = False):
    (Adam, {}),
    # (Adam, {'amsgrad': True}),
    # (Adam, {'maximize': True}),
    (Adam, {'foreach': False}),
    (Adam, {'foreach': True}),
    # (Adam, {'foreach': True, 'maximize': True}),
    # (Adam, {'foreach': True, 'amsgrad': True}),
    # (Adam, {'foreach': True, 'capturable': True}),
    (Adam, {'fused': True}),
    # (Adam, {'fused': True, 'amsgrad': True}),
    # (Adam, {'fused': True, 'maximize': True}),
    # (Adam, {'fused': True, 'capturable': True}),

    (AdamW, {}),
    # (AdamW, {'maximize': True}),
    (AdamW, {'foreach': False}),
    (AdamW, {'foreach': True}),
    # (AdamW, {'foreach': True, 'maximize': True, 'capturable': True}),
    (AdamW, {'fused': True}),
    # (AdamW, {'fused': True, 'amsgrad': True}),
    # (AdamW, {'fused': True, 'maximize': True}),
    # (AdamW, {'fused': True, 'capturable': True}),
    (Adamax, {}),
    # (Adamax, {'maximize': True}),
    (Adamax, {'foreach': False,}),
    (Adamax, {'foreach': True,}),
    # (Adamax, {'foreach': True, 'maximize': True}),
    (ASGD, {}),
    # (ASGD, {'maximize': True}),
    (ASGD, {'foreach': False,}),
    (ASGD, {'foreach': True,}),
    # (ASGD, {'foreach': True, 'maximize': True}),
    (SGD, {}),
    # (SGD, {'maximize': True}),
    (SGD, {'foreach': False,}),
    (SGD, {'foreach': True,}),
    # (SGD, {'foreach': True, 'momentum': 0.9, 'nesterov': True}),
    # (SGD, {'foreach': True, 'momentum': 0.9, }),
    # (SGD, {'foreach': True, 'maximize': True}),
    (RAdam, {}),
    (RAdam, {'foreach': False,}),
    (RAdam, {'foreach': True,}),
    (Rprop, {}),
    # (Rprop, {'maximize': True}),
    (Rprop, {'foreach': False,}),
    (Rprop, {'foreach': True,}),
    # (Rprop, {'foreach': True, 'maximize': True}),
    (RMSprop, {}),
    # (RMSprop, {'maximize': True}),
    (RMSprop, {'foreach': False,}),
    (RMSprop, {'foreach': True,}),
    # (RMSprop, {'foreach': True, 'maximize': True}),
    (NAdam, {}),
    (NAdam, {'foreach': False,}),
    (NAdam, {'foreach': True,}),
    (SparseAdam, {}),

    # LBFGS requires a closure
    # (torch.optim.LBFGS, {}),
]

devices = ['cuda', 'cpu']

def get_model_params(m) -> Any:
    model, _ = m.get_module()
    return model.parameters()

# This fakes a model forward & backward--we are not concerned about
# accuracy here, but about the perf of optim on particular shapes and
# dtypes of commonly used models!
def generate_random_gradients(parameters):
    for p in parameters:
        p.grad = torch.rand_like(p)

def optimizer_step(optimizer):
    optimizer.step()

def pt2_optimizer_step(optimizer):
    @torchdynamo.optimize("inductor")
    def f():
        optimizer.step()
    f()

def defaults_to_str(defaults: Dict[str, Any]) -> str:
    def entry_to_str(k, v) -> str:
        if isinstance(v, bool):
            return '' if not v else k
        return f'{k}={v}'
    return ', '.join([entry_to_str(k, v) for k, v in defaults.items()])

# fused/capturable requires params to be floats on CUDA
def defaults_require_cuda(defaults: Dict[str, Any]) -> bool:
    return 'fused' in defaults and defaults['fused'] or 'capturable' in defaults and defaults['capturable']

def run_model(modelName, device, Optim, defaults, maybe_pt2_):
    Model = load_model_by_name(modelName)   
    params = get_model_params(Model(device=device, test='train'))
    
    if Optim.__name__ == 'SGD':
        defaults['lr'] = 1e-2
    optim = Optim(params, **defaults)
    generate_random_gradients(params)

    return benchmark.Timer(
        stmt=f'{maybe_pt2_}optimizer_step(optim)',
        globals={'optim': optim, 'optimizer_step': optimizer_step, 'pt2_optimizer_step': pt2_optimizer_step},
        sub_label=f'{modelName}, {optim.__class__.__name__}, {device}',
        description=maybe_pt2_ + ' ' + ('default' if len(defaults) == 0 else defaults_to_str(defaults))
    ).blocked_autorange()


def run_benchmarks(optims: str, func_strs: List[str]) -> List[float]:
    results = []
    for mn, d, (O, defaults), func_str in itertools.product(MODEL_NAMES, devices, OPTIMIZERS, func_strs):
        if defaults_require_cuda(defaults) and d != 'cuda':
            continue
        if O.__name__ in optims:
            results.append(run_model(mn, d, O, defaults, func_str))
    return results


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optim", "-o",
        nargs="*",
        default=OPTIM_NAMES,
        choices=OPTIM_NAMES,
        help="List of optimizers to run tests on")
    parser.add_argument(
        "--funcs", "-f",
        nargs="*",
        default=FUNC_STRS,
        choices=FUNC_STRS,
        help="What optimizer.step() function variations to benchmark"
    )
    args = parser.parse_args(args)
    return args

# convert results into a JSON of description to mean time in seconds
def get_metrics(results: List[torch.utils.benchmark.utils.common.Measurement]) -> Dict[str, float]:
    metrics = {}
    for r in results:
        ts: torch.utils.benchmark.utils.common.TaskSpec = r.task_spec
        metrics[f'{ts.sub_label}, {ts.description}'] = r.mean
    return metrics

def run(args: List[str]):
    args = parse_args(args)
    results = run_benchmarks(args.optim, args.funcs)
    metrics: Dict[str, float] = get_metrics(results) 
    dump_output(BM_NAME, get_output_json(BM_NAME, metrics))
    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize(rowwise=True)
    compare.print()

if __name__ == '__main__':
    run(sys.argv[1:])
