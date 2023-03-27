from typing import Any, Dict, List, Tuple
from torchbenchmark import load_model_by_name
import torch
from torch import _dynamo as torchdynamo
from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, SGD, RAdam, Rprop, RMSprop, NAdam, SparseAdam, LBFGS
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
run_on_subset: bool = False

MODEL_NAMES: List[str] = list_models()
SUBSET_OF_MODEL_NAMES: List[str] = [
    'BERT_pytorch', 'DALLE2_pytorch', 'hf_GPT2_large', 'hf_T5_large', 'resnet50', 'timm_vision_transformer', 'yolov3'
]

DEVICES: str = ['cuda', 'cpu']

OPTIM_NAMES = [o.__name__ for o in [Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, SGD, RAdam, Rprop, RMSprop, NAdam, SparseAdam]]

FUNC_STRS = ['pt2_' , '']

OPTIMIZERS = [
    (Adadelta, {}),
    (Adadelta, {'maximize': True}),
    (Adadelta, {'foreach': False}),
    (Adadelta, {'differentiable': True}),
    (Adadelta, {'foreach': True}),
    (Adagrad, {}),
    (Adagrad, {'maximize': True}),
    (Adagrad, {'foreach': False}),
    (Adagrad, {'differentiable': True}),
    (Adagrad, {'foreach': True,}),
    (Adam, {}),
    (Adam, {'amsgrad': True, 'maximize': True}),
    (Adam, {'foreach': False}),
    (Adam, {'differentiable': True}),
    (Adam, {'foreach': True}),
    (Adam, {'foreach': True, 'maximize': True, 'capturable': True}),
    (Adam, {'foreach': True, 'maximize': True, 'capturable': True, 'amsgrad': True}),
    (Adam, {'fused': True}),
    (Adam, {'fused': True, 'amsgrad': True, 'maximize': True}),
    (Adam, {'fused': True, 'capturable': True}),
    (Adam, {'fused': True, 'capturable': True, 'amsgrad': True}),
    (AdamW, {}),
    (AdamW, {'amsgrad': True, 'maximize': True}),
    (AdamW, {'foreach': False}),
    (AdamW, {'differentiable': True}),
    (AdamW, {'foreach': True}),
    (AdamW, {'foreach': True, 'maximize': True, 'capturable': True}),
    (AdamW, {'foreach': True, 'maximize': True, 'capturable': True, 'amsgrad': True}),
    (AdamW, {'fused': True}),
    (AdamW, {'fused': True, 'amsgrad': True, 'maximize': True}),
    (AdamW, {'fused': True, 'capturable': True}),
    (AdamW, {'fused': True, 'capturable': True, 'amsgrad': True}),
    (Adamax, {}),
    (Adamax, {'maximize': True}),
    (Adamax, {'foreach': False}),
    (Adamax, {'differentiable': True}),
    (Adamax, {'foreach': True,}),
    (ASGD, {}),
    (ASGD, {'maximize': True}),
    (ASGD, {'foreach': False}),
    (ASGD, {'differentiable': True}),
    (ASGD, {'foreach': True}),
    (SGD, {}),
    (SGD, {'maximize': True}),
    (SGD, {'foreach': False}),
    (SGD, {'differentiable': True}),
    (SGD, {'foreach': True,}),
    (SGD, {'foreach': True, 'momentum': 0.9, 'nesterov': True}),
    (SGD, {'foreach': True, 'momentum': 0.9, }),
    (RAdam, {}),
    (RAdam, {'foreach': False}),
    (RAdam, {'differentiable': True}),
    (RAdam, {'foreach': True,}),
    (Rprop, {}),
    (Rprop, {'maximize': True}),
    (Rprop, {'foreach': False}),
    (Rprop, {'differentiable': True}),
    (Rprop, {'foreach': True}),
    (RMSprop, {}),
    (RMSprop, {'maximize': True}),
    (RMSprop, {'foreach': False}),
    (RMSprop, {'differentiable': True}),
    (RMSprop, {'foreach': True}),
    (NAdam, {}),
    (NAdam, {'foreach': False}),
    (NAdam, {'differentiable': True}),
    (NAdam, {'foreach': True}),
    (SparseAdam, {}),

    # LBFGS requires a closure
    # (LBFGS, {}),
]

DENSE_MODELS = [
    'BERT_pytorch',
    'Background_Matting',
    'DALLE2_pytorch',
    'LearningToPaint',
    'Super_SloMo',
    'alexnet',
    'attention_is_all_you_need_pytorch',
    'dcgan',
    'demucs',
    'densenet121',
    'detectron2_fasterrcnn_r_101_c4',
    'detectron2_fasterrcnn_r_101_dc5',
    'detectron2_fasterrcnn_r_101_fpn',
    'detectron2_fasterrcnn_r_50_c4',
    'detectron2_fasterrcnn_r_50_dc5',
    'detectron2_fasterrcnn_r_50_fpn',
    'detectron2_maskrcnn',
    'detectron2_maskrcnn_r_101_c4',
    'detectron2_maskrcnn_r_101_fpn',
    'detectron2_maskrcnn_r_50_c4',
    'detectron2_maskrcnn_r_50_fpn',
    'dlrm',
    'doctr_det_predictor',
    'doctr_reco_predictor',
    'drq',
    'fambench_xlmr',
    'fastNLP_Bert',
    'functorch_dp_cifar10',
    'functorch_maml_omniglot',
    'hf_Albert',
    'hf_Bart',
    'hf_Bert',
    'hf_Bert_large',
    'hf_BigBird',
    'hf_DistilBert',
    'hf_GPT2',
    'hf_GPT2_large',
    'hf_Longformer',
    'hf_Reformer',
    'hf_T5',
    'hf_T5_base',
    'hf_T5_large',
    'lennard_jones',
    'llama',
    'maml',
    'maml_omniglot',
    'mnasnet1_0',
    'mobilenet_v2',
    'mobilenet_v2_quantized_qat',
    'mobilenet_v3_large',
    'moco',
    'nvidia_deeprecommender',
    'opacus_cifar10',
    'phlippe_densenet',
    'phlippe_resnet',
    'pytorch_CycleGAN_and_pix2pix',
    'pytorch_stargan',
    'pytorch_struct',
    'pytorch_unet',
    'resnet152',
    'resnet18',
    'resnet50',
    'resnet50_quantized_qat',
    'resnext50_32x4d',
    'shufflenet_v2_x1_0',
    'soft_actor_critic',
    'speech_transformer',
    'squeezenet1_1',
    'tacotron2',
    'timm_efficientdet',
    'timm_efficientnet',
    'timm_nfnet',
    'timm_regnet',
    'timm_resnest',
    'timm_vision_transformer',
    'timm_vision_transformer_large',
    'timm_vovnet',
    'torchrec_dlrm',
    'tts_angular',
    'vgg16',
    'vision_maskrcnn',
    'yolov3'
]

# Skips! Exclusions are represented by a dictionary of incompatible configs, where
# optim => optimizer name
# model => model name
# func_str => func string (e.g., pt2_)
# device => device name
# defaults => list of flag descriptions (strings) to exclude, e.g. no_foreach
# Exclusions are general and will try to match on everything. For an exclusion
# {'optim': 'SparseAdam', 'model': 'BERT_pytorch'}, any configuration with
# SparseAdam on BERT_pytorch will be skipped.
EXCLUSIONS: List[Dict[str, Any]] = [
    # SparseAdam does not support dense gradients
    {'optim': 'SparseAdam', 'model': m} for m in DENSE_MODELS
] + [
    # torch.compile()'d optimizer.step() has too many arguments in C++
    # See GH issue: https://github.com/pytorch/pytorch/issues/97361
    {'model': m, 'device': 'cpu', 'func_str': 'pt2_', 'defaults': df} for m in [
        'BERT_pytorch', 'Background_Matting', 'Super_SloMo', 'attention_is_all_you_need_pytorch',
        'densenet121', 'detectron2_fasterrcnn_r_101_c4', 'detectron2_fasterrcnn_r_101_dc5',
        'detectron2_fasterrcnn_r_101_fpn', 'detectron2_fasterrcnn_r_50_fpn', 'detectron2_maskrcnn',
        'detectron2_maskrcnn_r_101_c4', 'detectron2_maskrcnn_r_101_fpn',
        'detectron2_maskrcnn_r_50_fpn', 'doctr_det_predictor', 'fambench_xlmr', 'fastNLP_Bert',
        'hf_Bart', 'hf_Bert', 'hf_Bert_large', 'hf_BigBird', 'hf_DistilBert', 'hf_GPT2',
        'hf_GPT2_large', 'hf_Longformer', 'hf_Reformer', 'hf_T5', 'hf_T5_base', 'hf_T5_large',
        'mnasnet1_0', 'mobilenet_v2', 'mobilenet_v2_quantized_qat', 'mobilenet_v3_large',
        'phlippe_densenet', 'pytorch_unet', 'resnet152', 'resnet50', 'resnet50_quantized_qat', 'resnext50_32x4d',
        'shufflenet_v2_x1_0', 'timm_efficientnet', 'timm_nfnet', 'timm_regnet',
        'timm_vision_transformer'
    ] for df in [[], ['differentiable']]
] + [
    # torch.compile()'d optimizer.step() has too many arguments in the generated
    # C++ kernel even when params are on CUDA for single tensor implementations.
    # See GH issue: https://github.com/pytorch/pytorch/issues/97361
    {'model': m, 'device': 'cuda', 'func_str': 'pt2_', 'defaults': ['no_foreach']} for m in [
        'DALLE2_pytorch', 'fambench_xlmr'] for df in ['no_foreach', 'differentiable']
] + [
    # torch.compile()'d optimizer.step() has too many arguments in the generated
    # C++ kernel even when params are on CUDA for single tensor implementations on NAdam.
    # See GH issue: https://github.com/pytorch/pytorch/issues/97361
    {'model': m, 'device': 'cuda', 'func_str': 'pt2_', 'defaults': [df], 'optim': 'NAdam'} for m in [
       'densenet121', 'doctr_reco_predictor', 'fambench_xlmr', 'hf_Bart', 'hf_Bert_large', 'hf_GPT2_large','hf_Longformer',
       'hf_T5_base', 'hf_T5_large', 'moco', 'resnet152'
    ] for df in ['no_foreach', 'differentiable']
] + [
    # torch.compile()'d optimizer.step() has too many arguments in the generated
    # C++ kernel even when params are on CUDA for single tensor implementations on ASGD.
    # See GH issue: https://github.com/pytorch/pytorch/issues/97361
    {'model': m, 'device': 'cuda', 'func_str': 'pt2_', 'defaults': [df], 'optim': 'ASGD'} for m in [
       'densenet121', 'fambench_xlmr', 'hf_Bart', 'hf_Bert_large', 'hf_GPT2_large', 'hf_Longformer',
       'hf_T5_base', 'hf_T5_large', 'moco'
    ] for df in ['no_foreach', 'differentiable']
] + [
    # DALL-E 2, timm_efficientdet, tacotron2 Not Supported on CPU
    {'model': 'DALLE2_pytorch', 'device': 'cpu'},
    {'model': 'tacotron2', 'device': 'cpu'},
    {'model': 'timm_efficientdet', 'device': 'cpu'},
    # FCOS train is not supported by upstream detectron2.
    # See GH issue: https://github.com/facebookresearch/detectron2/issues/4369.
    {'model': 'detectron2_fcos_r_50_fpn'},
    # moco uses DDP and DistributedDataParallel/allgather requires cuda
    {'model': 'moco', 'device': 'cpu'},
    # pyhpc_equation_of_state and pyhpc_isoneutral_mixing have no parameters
    {'model': 'pyhpc_equation_of_state'},
    {'model': 'pyhpc_isoneutral_mixing'},
    {'model': 'pyhpc_turbulent_kinetic_energy'},
    # fused/capturable requires params to be floats on CUDA
    {'defaults': ['fused'], 'device': 'cpu'},
    {'defaults': ['capturable'], 'device': 'cpu'},
] 

# Returns clones of params and not a generator.
def _get_model_params(m) -> List[torch.nn.Parameter]:
    model, _ = m.get_module()
    params_clone = []
    for p in model.parameters():
        params_clone.append(p.clone().detach())
    return params_clone

lil_cache: Tuple[str, str, List[torch.nn.Parameter]] = ('', '', [])

# Returns clones of params given a model name
def get_model_params(modelName: str, device: str) -> List[torch.nn.Parameter]:
    global lil_cache
    cached_mn, cached_d, cached_params = lil_cache
    if modelName == cached_mn and device == cached_d:
        return cached_params

    # free the old params before initializing a model to conserve memory
    lil_cache = ('', '', [])

    Model = load_model_by_name(modelName)
    try:
        # some (usually quantized) models do not support eval on CPU, but since we
        # only care about params + randomly generate grads, eval vs train doesn't matter
        params = _get_model_params(Model(device=device, test='train', batch_size=1))
    except NotImplementedError:
        try:
            params = _get_model_params(Model(device=device, test='eval', batch_size=1))
        except ValueError:
            params = _get_model_params(Model(device=device, test='eval'))
    except ValueError:
        params = _get_model_params(Model(device=device, test='train'))
    
    del Model
    
    lil_cache = (modelName, device, params)
    return params

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
    # We define lr for SGD, but we don't currently vary lr so it is effectively the default.
    defaults.pop('lr', None)
    if len(defaults) == 0:
        return 'default'

    def entry_to_str(k, v) -> str:
        if isinstance(v, bool):
            return 'no_' + k if not v else k
        return f'{k}={v}'
    return ', '.join([entry_to_str(k, v) for k, v in defaults.items()])

def is_excluded(mn: str, d: str, on: str, func_str: str, defaults: Dict[str, Any]) -> bool:
    return any([('model' not in e or e['model'] == mn) and
                ('device' not in e or e['device'] == d) and
                ('optim' not in e or e['optim'] == on) and
                ('funct_str' not in e or e['func_str'] == func_str) and
                ('defaults' not in e or all(f in defaults_to_str(defaults) for f in e['defaults'])) for e in EXCLUSIONS])
    
def run_model(modelName, device, Optim, defaults, maybe_pt2_):
    try:
        params = get_model_params(modelName, device)   
        print('getting params: ', params[0].size(), params[0].dtype, len(params), params[0].device)
        if Optim.__name__ == 'SGD':
            defaults['lr'] = 1e-2
        optim = Optim(params, **defaults)
        generate_random_gradients(params)
        pt2_description = '' if maybe_pt2_ == '' else '(pt2) '

        print(f'{datetime.datetime.now()}     {modelName}, {device}, {Optim}, {defaults_to_str(defaults)}, {maybe_pt2_}')
        return benchmark.Timer(
            stmt=f'{maybe_pt2_}optimizer_step(optim)',
            globals={'optim': optim, 'optimizer_step': optimizer_step, 'pt2_optimizer_step': pt2_optimizer_step},
            sub_label=f'{modelName}, {optim.__class__.__name__}, {device}',
            description=pt2_description + defaults_to_str(defaults),
        ).blocked_autorange()
    except Exception as e: 
        if not continue_on_error:
            raise e
        print(e)
        with open('errors.txt', 'a') as f:
            f.write(f'{datetime.datetime.now().timestamp()} {modelName}, {device}, {Optim}, {defaults_to_str(defaults)}, {maybe_pt2_}, {str(e)}\n')
        return None


def run_benchmarks(optims: List[str], func_strs: List[str], models: List[str], devices: List[str], flags: List[str]) -> List[float]:
    results = []
    optim_cfgs = [(O, defaults) for (O, defaults) in OPTIMIZERS if O.__name__ in optims and all(f in defaults_to_str(defaults) for f in flags)]

    if run_on_subset:
        models = SUBSET_OF_MODEL_NAMES
        optim_cfgs = [(O, defaults) for (O, defaults) in optim_cfgs if (all([x in ['foreach', 'fused', 'lr'] for x in defaults]))]
    
    for mn, d, (O, defaults), func_str in itertools.product(models, devices, optim_cfgs, func_strs):
        if (is_excluded(mn, d, O.__name__, func_str, defaults)):
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
        '--subset', '-s',
        action='store_true',
        help='Run benchmarks on a standard subset of models. Will overwrite the --models (-m) setting.'
    )
    parser.add_argument(
        '--devices', '-d',
        nargs='*',
        default=DEVICES,
        choices=DEVICES,
        help='List of devices to run tests on')
    parser.add_argument(
        '--default-flags', '-df',
        nargs='*',
        default=[],
        choices=['foreach', 'no_foreach', 'fused', 'maximize', 'capturable', 'differentiable', 'amsgrad', 'momentum', 'nesterov'],
        help='List of flag descriptions to run tests on. We serialize the configs to a string (see ' +
             'defaults_to_str()) and test for inclusion of the flag description in the string. ' +
             'For example, "foreach" will enable all default configs with "foreach", including ' +
             'those with other flags and also "no_foreach". Effectually, passing in more flags ' + 
             'will further limit the default configs run.\n'
    )
    parser.add_argument(
        '--continue-on-error', '-c',
        action='store_true',
        help='Continue running benchmarks on failure, errors will be written to errors.txt'
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
    global continue_on_error, run_on_subset
    continue_on_error = args.continue_on_error
    run_on_subset = args.subset
    results = run_benchmarks(args.optims, args.funcs, args.models, args.devices, args.default_flags)
    metrics: Dict[str, float] = get_metrics(results) 
    dump_output(BM_NAME, get_output_json(BM_NAME, metrics))
    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize(rowwise=True)
    compare.print()

if __name__ == '__main__':
    run(sys.argv[1:])
