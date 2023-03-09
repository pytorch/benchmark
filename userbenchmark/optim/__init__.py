from typing import Any, Dict, List, Tuple
from torchbenchmark import load_model_by_name
import torch
from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, SGD, RAdam, Rprop, RMSprop, NAdam, SparseAdam, LBFGS
import torch._dynamo as torchdynamo
import torch.utils.benchmark as benchmark
from userbenchmark.utils import REPO_PATH, add_path, dump_output, get_output_json
import argparse
import sys
import itertools
import datetime

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models


BM_NAME: str = 'optim'

continue_on_error: bool = False

MODEL_NAMES: List[str] = list_models()

DEVICES: str = ['cuda', 'cpu']

OPTIM_NAMES = [o.__name__ for o in [Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, SGD, RAdam, Rprop, RMSprop, NAdam, SparseAdam]]

FUNC_STRS = ['pt2_' , '']

OPTIMIZERS = [
    # Adadelta(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0, foreach: Optional[bool] = None,
    #          maximize: bool = False, differentiable: bool = False)
    (Adadelta, {}),
    (Adadelta, {'maximize': True}),
    (Adadelta, {'foreach': False,}),
    (Adadelta, {'foreach': True,}),
    (Adadelta, {'foreach': True, 'maximize': True}),
    # Adagrad(self, params, lr=1e-2, lr_decay=0, weight_decay=0, eps=1e-10, foreach: Optional[bool] = None, 
    #         maximize: bool = False, differentiable: bool = False)
    (Adagrad, {}),
    (Adagrad, {'maximize': True}),
    (Adagrad, {'foreach': False,}),
    (Adagrad, {'foreach': True,}),
    (Adagrad, {'foreach': True, 'maximize': True}),
    # Adam(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    #      weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
    #      maximize: bool = False, capturable: bool = False,
    #      differentiable: bool = False, fused: bool = False):
    (Adam, {}),
    (Adam, {'amsgrad': True}),
    (Adam, {'maximize': True}),
    (Adam, {'foreach': False}),
    (Adam, {'differentiable': True}),
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
    (AdamW, {'foreach': False}),
    (AdamW, {'foreach': True}),
    (AdamW, {'foreach': True, 'maximize': True, 'capturable': True}),
    (AdamW, {'fused': True}),
    (AdamW, {'fused': True, 'amsgrad': True}),
    (AdamW, {'fused': True, 'maximize': True}),
    (AdamW, {'fused': True, 'capturable': True}),
    (Adamax, {}),
    (Adamax, {'maximize': True}),
    (Adamax, {'foreach': False,}),
    (Adamax, {'foreach': True,}),
    (Adamax, {'foreach': True, 'maximize': True}),
    (ASGD, {}),
    (ASGD, {'maximize': True}),
    (ASGD, {'foreach': False,}),
    (ASGD, {'foreach': True,}),
    (ASGD, {'foreach': True, 'maximize': True}),
    (SGD, {}),
    (SGD, {'maximize': True}),
    (SGD, {'foreach': False,}),
    (SGD, {'foreach': True,}),
    (SGD, {'foreach': True, 'momentum': 0.9, 'nesterov': True}),
    (SGD, {'foreach': True, 'momentum': 0.9, }),
    (SGD, {'foreach': True, 'maximize': True}),
    (RAdam, {}),
    (RAdam, {'foreach': False,}),
    (RAdam, {'foreach': True,}),
    (Rprop, {}),
    (Rprop, {'maximize': True}),
    (Rprop, {'foreach': False,}),
    (Rprop, {'foreach': True,}),
    (Rprop, {'foreach': True, 'maximize': True}),
    (RMSprop, {}),
    (RMSprop, {'maximize': True}),
    (RMSprop, {'foreach': False,}),
    (RMSprop, {'foreach': True,}),
    (RMSprop, {'foreach': True, 'maximize': True}),
    (NAdam, {}),
    (NAdam, {'foreach': False,}),
    (NAdam, {'foreach': True,}),
    (SparseAdam, {}),

    # LBFGS requires a closure
    # (torch.optim.LBFGS, {}),
]

DENSE_MODELS = set()
"""
collected so far, but we should have something else collect it for us
               ['BERT_pytorch', 'Background_Matting', 'DALLE2_pytorch', 'LearningToPaint', 'Super_SloMo', 'alexnet',
                'attention_is_all_you_need_pytorch', 'dcgan', 'demucs', 'densenet121', 'detectron2_fasterrcnn_r_101_c4',
                'detectron2_fasterrcnn_r_101_dc5', 'detectron2_fasterrcnn_r_101_fpn', 'detectron2_fasterrcnn_r_50_c4',
                'detectron2_fasterrcnn_r_50_dc5', 'detectron2_fasterrcnn_r_50_fpn', 'detectron2_maskrcnn',
                'detectron2_maskrcnn_r_101_c4', 'detectron2_maskrcnn_r_101_fpn', 'detectron2_maskrcnn_r_50_c4',
                'detectron2_maskrcnn_r_50_fpn', 'fambench_xlmr', 'fastNLP_Bert', 'functorch_dp_cifar10']

"""
# Skips! Exclusions are represented by a dictionary of incompatible configs, where
# optim => optimizer name
# model => model name
# func_str => func string (e.g., pt2_)
# device => device name
# Exclusions are general and will try to match on everything. For an exclusion
# {'optim': 'SparseAdam', 'model': 'BERT_pytorch'}, any configuration with
# SparseAdam on BERT_pytorch will be skipped.
EXCLUSIONS: List[Dict[str, Any]] = [
    # SparseAdam does not support dense gradients
    {'optim': 'SparseAdam', 'model': m} for m in DENSE_MODELS
] + [
    # DALL-E 2, timm_efficientdet Not Supported on CPU
    {'model': 'DALLE2_pytorch', 'device': 'cpu'},
    {'model': 'timm_efficientdet', 'device': 'cpu'},
    # FCOS train is not supported by upstream detectron2.
    # See GH Issue: https://github.com/facebookresearch/detectron2/issues/4369.
    {'model': 'detectron2_fcos_r_50_fpn'},
    # moco uses DDP and DistributedDataParallel/allgather requires cuda
    {'model': 'moco', 'device': 'cpu'},
    # pyhpc_equation_of_state and pyhpc_isoneutral_mixing have no parameters
    {'model': 'pyhpc_equation_of_state'},
    {'model': 'pyhpc_isoneutral_mixing'},
    {'model': 'pyhpc_turbulent_kinetic_energy'},
] 

# Returns clones of params and not a generator.
def get_model_params(m) -> Any:
    model, _ = m.get_module()
    params_clone = []
    for p in model.parameters():
        params_clone.append(p.clone().detach())
    return params_clone

# This fakes a model forward & backward--we are not concerned about
# accuracy here, but about the perf of optim on particular shapes and
# dtypes of commonly used models!
def generate_random_gradients(parameters):
    for p in parameters:
        p.grad = torch.rand_like(p)

def optimizer_step(optimizer):
    optimizer.step()

def pt2_optimizer_step(optimizer):
    @torchdynamo.optimize('inductor')
    def f():
        optimizer.step()
    f()

def defaults_to_str(defaults: Dict[str, Any]) -> str:
    def entry_to_str(k, v) -> str:
        if isinstance(v, bool):
            return 'no_' + k if not v else k
        return f'{k}={v}'
    return ', '.join([entry_to_str(k, v) for k, v in defaults.items()])

# fused/capturable requires params to be floats on CUDA
def defaults_require_cuda(defaults: Dict[str, Any]) -> bool:
    return 'fused' in defaults and defaults['fused'] or 'capturable' in defaults and defaults['capturable']

def is_excluded(mn: str, d: str, on: str, func_str: str) -> bool:
    return any([('model' not in e or e['model'] == mn) and
                ('device' not in e or e['device'] == d) and
                ('optim' not in e or e['optim'] == on) and
                ('funct_str' not in e or e['func_str'] == func_str) for e in EXCLUSIONS])
    
def run_model(modelName, device, Optim, defaults, maybe_pt2_):
    try:
        Model = load_model_by_name(modelName)   
        try: 
            params = get_model_params(Model(device=device, test='train'))
        except NotImplementedError:
            params = get_model_params(Model(device=device, test='eval'))
        if Optim.__name__ == 'SGD':
            defaults['lr'] = 1e-2
        if len(params) > 0 and params[0].layout == torch.strided and 'Sparse' in Optim.__name__:
            # don't run Sparse optimizers on dense gradients
            DENSE_MODELS.add(modelName)
            return None
        optim = Optim(params, **defaults)
        generate_random_gradients(params)
        pt2_description = '' if maybe_pt2_ == '' else '(pt2) '

        print(f'{datetime.datetime.now()}     {modelName}, {device}, {Optim}, {defaults_to_str(defaults)}, {maybe_pt2_}')

        return benchmark.Timer(
            stmt=f'{maybe_pt2_}optimizer_step(optim)',
            globals={'optim': optim, 'optimizer_step': optimizer_step, 'pt2_optimizer_step': pt2_optimizer_step},
            sub_label=f'{modelName}, {optim.__class__.__name__}, {device}',
            description=pt2_description + ('default' if len(defaults) == 0 else defaults_to_str(defaults))
        ).blocked_autorange()
    except Exception as e: 
        if not continue_on_error:
            raise e
        with open('errors.txt', 'a') as f:
            f.write(f'{datetime.datetime.now()} {modelName}, {device}, {Optim}, {defaults_to_str(defaults)}, {maybe_pt2_}, {str(e)}\n')
        return None


def run_benchmarks(optims: List[str], func_strs: List[str], models: List[str], devices: List[str]) -> List[float]:
    results = []
    optim_cfgs = [(O, defaults) for (O, defaults) in OPTIMIZERS if O.__name__ in optims]
    for mn, d, (O, defaults), func_str in itertools.product(models, devices, optim_cfgs, func_strs):
        if is_excluded(mn, d, O.__name__, func_str) or (defaults_require_cuda(defaults) and d != 'cuda'):
            continue
        bm = run_model(mn, d, O, defaults, func_str)
        if bm is not None:
            results.append(bm)
    return results


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--optims', '-o',
        nargs='*',
        default=OPTIM_NAMES,
        choices=OPTIM_NAMES,
        help='List of optimizers to run tests on')
    parser.add_argument(
        '--funcs', '-f',
        nargs='*',
        default=FUNC_STRS,
        choices=FUNC_STRS,
        help='What optimizer.step() function variations to benchmark'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='*',
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help='List of models to run tests on')
    parser.add_argument(
        '--devices', '-d',
        nargs='*',
        default=DEVICES,
        choices=DEVICES,
        help='List of devices to run tests on')
    parser.add_argument(
        '--continue-on-error', '-c',
        action='store_true'
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
    global continue_on_error
    continue_on_error = args.continue_on_error
    results = run_benchmarks(args.optims, args.funcs, args.models, args.devices)
    metrics: Dict[str, float] = get_metrics(results) 
    dump_output(BM_NAME, get_output_json(BM_NAME, metrics))
    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize(rowwise=True)
    compare.print()
    import json
    with open('errors.txt', 'a') as f:
        json.dump(list(DENSE_MODELS), f, indent=4)

if __name__ == '__main__':
    run(sys.argv[1:])
