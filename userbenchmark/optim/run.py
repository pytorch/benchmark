import argparse
import datetime
import itertools
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.utils.benchmark as benchmark
import yaml
from torch import _dynamo as torchdynamo
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam,
)
from torchbenchmark import load_model_by_name
from userbenchmark.utils import add_path, dump_output, get_output_json, REPO_PATH

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models


BM_NAME: str = "optim"

continue_on_error: bool = False
run_on_subset: bool = False
run_basic_configs: bool = False
ignore_skips: bool = False


# Models that are unstable in torch-nightly should not run in the optim world either
def get_unstable_models() -> Set[str]:
    unstable_models: Set[str] = set()
    yaml_file_path = REPO_PATH.joinpath(
        "userbenchmark/torch-nightly/v3-cuda-tests.yaml"
    )
    with open(yaml_file_path, "r") as yf:
        config_obj = yaml.safe_load(yf)
    for d in config_obj["cuda"]:
        if not d["stable"]:
            unstable_models.add(d["model"])
    return unstable_models


unstable_models: Set[str] = get_unstable_models()

MODEL_NAMES: List[str] = list_models()
SUBSET_OF_MODEL_NAMES: List[str] = [
    "BERT_pytorch",
    "DALLE2_pytorch",
    "hf_GPT2_large",
    "hf_T5_large",
    "resnet50",
    "timm_vision_transformer_large",
    "yolov3",
]

# PT2 compilation can take up 4 minutes for 1000 parameters even for foreach implementations,
# and erroring has not been very diverse across models, so we pick one small model and one
# larger model to ascertain PT2 performance over time. We skip everything else.
# We _are_ working on minimizing compilation for foreach implementations.
#
# PT2 dynamo tracing for the for-loop implementation takes over 30s.
# This is known + NOT going to be improved anytime soon, see
# https://github.com/pytorch/torchdynamo/issues/1803#issuecomment-1336688894
MODELS_TO_RUN_ON_PT2: List[str] = ["resnet18", "timm_vision_transformer_large"]

# NOTE: While it is possible to run these benchmarks on CPU, we skip running on CPU in CI because CPU stats can be
# unstable and we had stopped reporting them. You'll still be able to use this script to run CPU though, as it may
# be useful as a more local comparison point for implementations like forloop.
DEVICES: List[str] = ["cuda", "cpu"]

OPTIM_NAMES = [
    o.__name__
    for o in [
        Adadelta,
        Adagrad,
        Adam,
        AdamW,
        Adamax,
        ASGD,
        SGD,
        RAdam,
        Rprop,
        RMSprop,
        NAdam,
        SparseAdam,
    ]
]

FUNC_STRS = ["", "pt2_"]

OPTIMIZERS = [
    (Adadelta, {}),
    (Adadelta, {"maximize": True}),
    (Adadelta, {"foreach": False}),
    (Adadelta, {"differentiable": True}),
    (Adadelta, {"foreach": True}),
    (Adagrad, {}),
    (Adagrad, {"maximize": True}),
    (Adagrad, {"foreach": False}),
    (Adagrad, {"differentiable": True}),
    (
        Adagrad,
        {
            "foreach": True,
        },
    ),
    (Adam, {}),
    (Adam, {"amsgrad": True, "maximize": True}),
    (Adam, {"foreach": False}),
    (Adam, {"differentiable": True}),
    (Adam, {"foreach": True}),
    (Adam, {"foreach": True, "maximize": True, "capturable": True}),
    (Adam, {"foreach": True, "maximize": True, "capturable": True, "amsgrad": True}),
    (Adam, {"fused": True}),
    (Adam, {"fused": True, "amsgrad": True, "maximize": True}),
    (Adam, {"fused": True, "capturable": True}),
    (Adam, {"fused": True, "capturable": True, "amsgrad": True}),
    (AdamW, {}),
    (AdamW, {"amsgrad": True, "maximize": True}),
    (AdamW, {"foreach": False}),
    (AdamW, {"differentiable": True}),
    (AdamW, {"foreach": True}),
    (AdamW, {"foreach": True, "maximize": True, "capturable": True}),
    (AdamW, {"foreach": True, "maximize": True, "capturable": True, "amsgrad": True}),
    (AdamW, {"fused": True}),
    (AdamW, {"fused": True, "amsgrad": True, "maximize": True}),
    (AdamW, {"fused": True, "capturable": True}),
    (AdamW, {"fused": True, "capturable": True, "amsgrad": True}),
    (Adamax, {}),
    (Adamax, {"maximize": True}),
    (Adamax, {"foreach": False}),
    (Adamax, {"differentiable": True}),
    (
        Adamax,
        {
            "foreach": True,
        },
    ),
    (ASGD, {}),
    (ASGD, {"maximize": True}),
    (ASGD, {"foreach": False}),
    (ASGD, {"differentiable": True}),
    (ASGD, {"foreach": True}),
    (SGD, {}),
    (SGD, {"maximize": True}),
    (SGD, {"foreach": False}),
    (SGD, {"differentiable": True}),
    (
        SGD,
        {
            "foreach": True,
        },
    ),
    (SGD, {"foreach": True, "momentum": 0.9, "nesterov": True}),
    (
        SGD,
        {
            "foreach": True,
            "momentum": 0.9,
        },
    ),
    (
        SGD,
        {
            "fused": True,
        },
    ),
    (
        SGD,
        {
            "fused": True,
            "momentum": 0.9,
        },
    ),
    (RAdam, {}),
    (RAdam, {"foreach": False}),
    (RAdam, {"differentiable": True}),
    (
        RAdam,
        {
            "foreach": True,
        },
    ),
    (Rprop, {}),
    (Rprop, {"maximize": True}),
    (Rprop, {"foreach": False}),
    (Rprop, {"differentiable": True}),
    (Rprop, {"foreach": True}),
    (RMSprop, {}),
    (RMSprop, {"maximize": True}),
    (RMSprop, {"foreach": False}),
    (RMSprop, {"differentiable": True}),
    (RMSprop, {"foreach": True}),
    (NAdam, {}),
    (NAdam, {"foreach": False}),
    (NAdam, {"differentiable": True}),
    (NAdam, {"foreach": True}),
    (SparseAdam, {}),
    # LBFGS requires a closure
    # (LBFGS, {}),
]

DENSE_MODELS = [
    "BERT_pytorch",
    "Background_Matting",
    "DALLE2_pytorch",
    "LearningToPaint",
    "Super_SloMo",
    "alexnet",
    "basic_gnn_edgecnn",
    "basic_gnn_gcn",
    "basic_gnn_gin",
    "basic_gnn_sage",
    "cm3leon_generate",
    "dcgan",
    "demucs",
    "densenet121",
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "detectron2_maskrcnn",
    "detectron2_maskrcnn_r_101_c4",
    "detectron2_maskrcnn_r_101_fpn",
    "detectron2_maskrcnn_r_50_c4",
    "detectron2_maskrcnn_r_50_fpn",
    "dlrm",
    "doctr_det_predictor",
    "doctr_reco_predictor",
    "drq",
    "fambench_xlmr",
    "fastNLP_Bert",
    "functorch_dp_cifar10",
    "functorch_maml_omniglot",
    "gat",
    "gcn",
    "hf_Albert",
    "hf_Bart",
    "hf_Bert",
    "hf_Bert_large",
    "hf_BigBird",
    "hf_DistilBert",
    "hf_GPT2",
    "hf_GPT2_large",
    "hf_Longformer",
    "hf_Reformer",
    "hf_T5",
    "hf_T5_base",
    "hf_T5_generate",
    "hf_T5_large",
    "hf_Whisper",
    "hf_clip",
    "lennard_jones",
    "llama",
    "llama_v2_7b_16h",
    "maml",
    "maml_omniglot",
    "mnasnet1_0",
    "mobilenet_v2",
    "mobilenet_v2_quantized_qat",
    "mobilenet_v3_large",
    "moco",
    "nanogpt",
    "nvidia_deeprecommender",
    "opacus_cifar10",
    "phlippe_densenet",
    "phlippe_resnet",
    "pytorch_CycleGAN_and_pix2pix",
    "pytorch_stargan",
    "pytorch_unet",
    "resnet152",
    "resnet18",
    "resnet50",
    "resnet50_quantized_qat",
    "resnext50_32x4d",
    "sage",
    "sam",
    "shufflenet_v2_x1_0",
    "simple_gpt",
    "soft_actor_critic",
    "speech_transformer",
    "squeezenet1_1",
    "stable_diffusion",
    "stable_diffusion_text_encoder",
    "stable_diffusion_unet",
    "tacotron2",
    "timm_efficientdet",
    "timm_efficientnet",
    "timm_nfnet",
    "timm_regnet",
    "timm_resnest",
    "timm_vision_transformer",
    "timm_vision_transformer_large",
    "timm_vovnet",
    "torchrec_dlrm",
    "tts_angular",
    "vgg16",
    "vision_maskrcnn",
    "yolov3",
]

# Skips! Exclusions are represented by a dictionary of incompatible configs, where
# optim => optimizer name
# model => model name
# func_str => func string (e.g., pt2_)
# device => device name
# defaults => list of flag descriptions (strings) to exclude, e.g. no_foreach
#             if empty list, will exclude all configurations
# Exclusions are general and will try to match on everything. For an exclusion
# {'optim': 'SparseAdam', 'model': 'BERT_pytorch'}, any configuration with
# SparseAdam on BERT_pytorch will be skipped.
EXCLUSIONS: List[Dict[str, Any]] = (
    [
        # Skip models deemed unstable by torch-nightly
        {"model": m}
        for m in unstable_models
    ]
    + [
        # 16h currently OOMs, but once it supports train, we should remove this line
        # See tracker https://github.com/pytorch/benchmark/issues/1793
        {"model": "llama_v2_7b_16h"}
    ]
    + [
        # Model needs to be run via dynamo torchbench and be provided distributed parameters
        {"model": "simple_gpt"}
    ]
    + [
        # SparseAdam does not support dense gradients
        {"optim": "SparseAdam", "model": m}
        for m in DENSE_MODELS
    ]
    + [
        # DALL-E 2, timm_efficientdet, tacotron2 Not Supported on CPU
        {"model": "DALLE2_pytorch", "device": "cpu"},
        {"model": "tacotron2", "device": "cpu"},
        {"model": "timm_efficientdet", "device": "cpu"},
        # FCOS train is not supported by upstream detectron2.
        # See GH issue: https://github.com/facebookresearch/detectron2/issues/4369.
        {"model": "detectron2_fcos_r_50_fpn"},
        # moco uses DDP and DistributedDataParallel/allgather requires cuda
        {"model": "moco", "device": "cpu"},
        # pyhpc_equation_of_state and pyhpc_isoneutral_mixing have no parameters
        {"model": "pyhpc_equation_of_state"},
        {"model": "pyhpc_isoneutral_mixing"},
        {"model": "pyhpc_turbulent_kinetic_energy"},
        # fused/capturable requires params to be floats on CUDA
        {"defaults": ["fused"], "device": "cpu"},
        {"defaults": ["capturable"], "device": "cpu"},
    ]
    + [
        # PT2 compilation takes too long, so we only enable PT2 on a tiny subset of models.
        # See note above on MODELS_TO_RUN_ON_PT2.
        {"model": m, "device": d, "func_str": "pt2_", "defaults": []}
        for d in DEVICES
        for m in set(MODEL_NAMES) - set(MODELS_TO_RUN_ON_PT2)
    ]
    + [
        {"func_str": "pt2_", "defaults": [df]}
        for df in ["maximize", "differentiable", "capturable", "amsgrad"]
    ]
    + [
        # torch.compile()'d optimizer.step() has too many arguments in C++
        # See GH issue: https://github.com/pytorch/pytorch/issues/97361
        {"model": m, "device": "cpu", "func_str": "pt2_", "defaults": []}
        for m in [
            "BERT_pytorch",
            "Background_Matting",
            "Super_SloMo",
            "densenet121",
            "detectron2_fasterrcnn_r_101_c4",
            "detectron2_fasterrcnn_r_101_dc5",
            "detectron2_fasterrcnn_r_101_fpn",
            "detectron2_fasterrcnn_r_50_fpn",
            "detectron2_maskrcnn",
            "detectron2_maskrcnn_r_101_c4",
            "detectron2_maskrcnn_r_101_fpn",
            "detectron2_maskrcnn_r_50_fpn",
            "doctr_det_predictor",
            "doctr_reco_predictor",
            "fambench_xlmr",
            "fastNLP_Bert",
            "hf_Bart",
            "hf_Bert",
            "hf_Bert_large",
            "hf_BigBird",
            "hf_DistilBert",
            "hf_GPT2",
            "hf_GPT2_large",
            "hf_Longformer",
            "hf_Reformer",
            "hf_T5",
            "hf_T5_base",
            "hf_T5_large",
            "llama",
            "mnasnet1_0",
            "mobilenet_v2",
            "mobilenet_v2_quantized_qat",
            "mobilenet_v3_large",
            "phlippe_densenet",
            "pytorch_unet",
            "resnet152",
            "resnet50",
            "resnet50_quantized_qat",
            "resnext50_32x4d",
            "shufflenet_v2_x1_0",
            "timm_efficientnet",
            "timm_nfnet",
            "timm_regnet",
            "timm_vision_transformer",
            "yolov3",
        ]
    ]
    + [
        # torch.compile()'d optimizer.step() has too many arguments in the generated
        # C++ kernel for both CUDA and CPU for single tensor implementations.
        # See GH issue: https://github.com/pytorch/pytorch/issues/97361
        {"model": m, "func_str": "pt2_", "defaults": [df]}
        for m in ["DALLE2_pytorch", "fambench_xlmr"]
        for df in ["no_foreach", "differentiable"]
    ]
    + [
        # torch.compile()'d optimizer.step() has too many arguments in the generated
        # C++ kernel even when params are on CUDA for single tensor implementations on NAdam.
        # See GH issue: https://github.com/pytorch/pytorch/issues/97361
        {
            "model": m,
            "device": "cuda",
            "func_str": "pt2_",
            "defaults": [df],
            "optim": "NAdam",
        }
        for m in [
            "densenet121",
            "doctr_reco_predictor",
            "fambench_xlmr",
            "hf_Bart",
            "hf_Bert_large",
            "hf_GPT2_large",
            "hf_Longformer",
            "hf_T5_base",
            "hf_T5_large",
            "moco",
            "resnet152",
            "timm_vision_transformer",
            "timm_vision_transformer_large",
            "yolov3",
        ]
        for df in ["no_foreach", "differentiable"]
    ]
    + [
        # torch.compile()'d optimizer.step() has too many arguments in the generated
        # C++ kernel even when params are on CUDA for single tensor implementations on ASGD.
        # See GH issue: https://github.com/pytorch/pytorch/issues/97361
        {
            "model": m,
            "device": "cuda",
            "func_str": "pt2_",
            "defaults": [df],
            "optim": "ASGD",
        }
        for m in [
            "densenet121",
            "fambench_xlmr",
            "hf_Bart",
            "hf_Bert_large",
            "hf_GPT2_large",
            "hf_Longformer",
            "hf_T5_base",
            "hf_T5_large",
            "moco",
        ]
        for df in ["no_foreach", "differentiable"]
    ]
)


# Returns clones of params and not a generator.
def _get_model_params(m) -> List[torch.nn.Parameter]:
    model, _ = m.get_module()
    params_clone = []
    for p in model.parameters():
        params_clone.append(p.clone().detach())
    return params_clone


lil_cache: Tuple[str, str, List[torch.nn.Parameter]] = ("", "", [])


# Returns clones of params given a model name
def get_model_params(modelName: str, device: str) -> List[torch.nn.Parameter]:
    global lil_cache
    cached_mn, cached_d, cached_params = lil_cache
    if modelName == cached_mn and device == cached_d:
        return cached_params

    # free the old params before initializing a model to conserve memory
    lil_cache = ("", "", [])
    torch.cuda.empty_cache()

    Model = load_model_by_name(modelName)

    # some (usually quantized) models do not support eval on CPU, but since we
    # only care about params + randomly generate grads, eval vs train doesn't matter
    try:
        params = _get_model_params(Model(device=device, test="train", batch_size=1))
    except:
        try:
            params = _get_model_params(Model(device=device, test="eval", batch_size=1))
        except:
            try:
                params = _get_model_params(Model(device=device, test="train"))
            except:
                params = _get_model_params(Model(device=device, test="eval"))
    finally:
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
    @torchdynamo.optimize("inductor")
    def f():
        optimizer.step()

    f()


def defaults_to_str(defaults: Dict[str, Any]) -> str:
    # We define lr for SGD, but we don't currently vary lr so it is effectively the default.
    defaults.pop("lr", None)
    if len(defaults) == 0:
        return "default"

    def entry_to_str(k, v) -> str:
        if isinstance(v, bool):
            return "no_" + k if not v else k
        return f"{k}={v}"

    return ", ".join([entry_to_str(k, v) for k, v in defaults.items()])


def is_excluded(
    mn: str, d: str, on: str, func_str: str, defaults: Dict[str, Any]
) -> bool:
    return any(
        [
            ("model" not in e or e["model"] == mn)
            and ("device" not in e or e["device"] == d)
            and ("optim" not in e or e["optim"] == on)
            and ("func_str" not in e or e["func_str"] == func_str)
            and (
                "defaults" not in e
                or all(f in defaults_to_str(defaults) for f in e["defaults"])
            )
            for e in EXCLUSIONS
        ]
    )


def run_model(modelName, device, Optim, defaults, maybe_pt2_):
    try:
        params = get_model_params(modelName, device)
        print(
            datetime.datetime.now(),
            "getting params: ",
            params[0].size(),
            params[0].dtype,
            len(params),
            params[0].device,
        )
        optim = Optim(params, **defaults)
        generate_random_gradients(params)
        pt2_description = "" if maybe_pt2_ == "" else "(pt2) "

        print(
            f"{datetime.datetime.now()}     python -m userbenchmark.optim.run -m {modelName} -d {device}"
            + f' -o {Optim.__name__} --df "{defaults_to_str(defaults)}" -f {maybe_pt2_}'
        )

        compile_r = None
        sub_label = f"{modelName}, {optim.__class__.__name__}, {device}"
        description = pt2_description + defaults_to_str(defaults)

        # Get compile time by running 5 times and subtracting
        #   first entry - avg(entries 3 through 5)
        # skipping the second entry due to high variance of first cache hit
        if maybe_pt2_ != "":
            times = []
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            for _ in range(5):
                t0 = time.perf_counter()
                pt2_optimizer_step(optim)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            compile_r = (
                f"compile_time, {sub_label}, {description}",
                times[0] - sum(times[2:5]) / 3,
            )

        r = benchmark.Timer(
            stmt=f"{maybe_pt2_}optimizer_step(optim)",
            globals={
                "optim": optim,
                "optimizer_step": optimizer_step,
                "pt2_optimizer_step": pt2_optimizer_step,
            },
            sub_label=sub_label,
            description=description,
        ).blocked_autorange()

        return r, compile_r
    except Exception as e:
        if not continue_on_error:
            raise e
        print(e)
        with open("errors.txt", "a") as f:
            f.write(
                f"{datetime.datetime.now()} python -m userbenchmark.optim.run -m {modelName} -d {device}"
                + f' -o {Optim.__name__} --df "{defaults_to_str(defaults)}" -f {maybe_pt2_}{str(e)}\n'
            )
        return None, None


def run_benchmarks(
    optims: List[str],
    func_strs: List[str],
    models: List[str],
    devices: List[str],
    flags: List[str],
) -> Tuple[List[torch.utils.benchmark.utils.common.Measurement], Dict[str, float]]:
    results = []
    compile_metrics = {}
    optim_cfgs = [
        (O, defaults)
        for (O, defaults) in OPTIMIZERS
        if O.__name__ in optims and all(f in defaults_to_str(defaults) for f in flags)
    ]

    if run_on_subset:
        models = [m for m in SUBSET_OF_MODEL_NAMES if m in models]

    if run_on_subset or run_basic_configs:
        optim_cfgs = [
            (O, defaults)
            for (O, defaults) in optim_cfgs
            if (all([x in ["foreach", "fused", "lr"] for x in defaults]))
        ]

    for mn, d, (O, defaults), func_str in itertools.product(
        models, devices, optim_cfgs, func_strs
    ):
        if not ignore_skips and is_excluded(mn, d, O.__name__, func_str, defaults):
            continue
        r, compile_r = run_model(mn, d, O, defaults, func_str)
        if r is not None:
            results.append(r)
        if compile_r is not None:
            metric_name, compile_time = compile_r
            compile_metrics[metric_name] = compile_time
    return results, compile_metrics


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optims",
        "-o",
        nargs="*",
        default=OPTIM_NAMES,
        choices=OPTIM_NAMES,
        help="List of optimizers to run tests on",
    )
    parser.add_argument(
        "--funcs",
        "-f",
        nargs="*",
        default=FUNC_STRS,
        choices=FUNC_STRS,
        help="What optimizer.step() function variations to benchmark. NOTE: there is an underscore "
        + 'for "pt2_"!',
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="*",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help="List of models to run tests on",
    )
    parser.add_argument(
        "--subset",
        "-s",
        action="store_true",
        help="Run benchmarks on a standard subset of models. If the --models (-m) is set, we will "
        + "take the intersection of the requested models and the defined subset. For example, "
        + "`...-s -m llama yolov3` will ONLY run yolov3.",
    )
    parser.add_argument(
        "--basic",
        "-b",
        action="store_true",
        help="Run benchmarks on a standard subset of optimizer configs. If the --defaults-flags (--df)"
        + " is set, we will take the intersection of the requested configs and the defined subset. "
        + "For example, `...-b --df maximize fused` will ONLY run fused.",
    )
    parser.add_argument(
        "--devices",
        "-d",
        nargs="*",
        default=DEVICES,
        choices=DEVICES,
        help="List of devices to run tests on",
    )
    parser.add_argument(
        "--default-flags",
        "--df",
        nargs="*",
        default=[],
        help="List of flag descriptions to run tests on. We serialize the configs to a string (see "
        + "defaults_to_str()) and test for inclusion of the flag description in the string. "
        + 'For example, "foreach" will enable all default configs with "foreach", including '
        + 'those with other flags and also "no_foreach". Effectually, passing in more flags '
        + "will further limit the default configs run.\nValid flags include: foreach, no_foreach, "
        + "fused, maximize, capturable, differentiable, default, amsgrad, momentum, nesterov"
        + " and more!\n",
    )
    parser.add_argument(
        "--continue-on-error",
        "-c",
        action="store_true",
        help="Continue running benchmarks on failure, errors will be written to errors.txt",
    )
    parser.add_argument(
        "--output-dir",
        "--od",
        default=None,
        type=str,
        help='name of directory path in which to dump the metrics json, e.g., "./.userbenchmark/optim/tmp". '
        + 'If None, we will dump output the metrics json to "REPO_ROOT/.userbenchmark/optim".',
    )
    parser.add_argument(
        "--ignore-skips",
        "-i",
        action="store_true",
        help="Runs ALL benchmarks ignoring any skips. This allows for easy testing of current skipped "
        + "benchmarks once one believes they should be fixed. Beware though! You may run into errors "
        + "that were previously hidden by the exclusions.",
    )
    args = parser.parse_args(args)
    return args


# convert results into a JSON of description to mean time in seconds
def get_metrics(
    results: List[torch.utils.benchmark.utils.common.Measurement],
) -> Dict[str, float]:
    metrics = {}
    for r in results:
        ts: torch.utils.benchmark.utils.common.TaskSpec = r.task_spec
        metrics[f"{ts.sub_label}, {ts.description}"] = r.mean
    return metrics


def run(args: List[str]):
    args = parse_args(args)
    global continue_on_error, run_on_subset, run_basic_configs, ignore_skips
    continue_on_error = args.continue_on_error
    run_on_subset = args.subset
    run_basic_configs = args.basic
    ignore_skips = args.ignore_skips
    target_dir = Path(args.output_dir) if args.output_dir is not None else None
    if target_dir is not None:
        target_dir.mkdir(exist_ok=True, parents=True)

    results, compile_metrics = run_benchmarks(
        args.optims, args.funcs, args.models, args.devices, args.default_flags
    )
    metrics: Dict[str, float] = get_metrics(results)
    dump_output(
        BM_NAME,
        get_output_json(BM_NAME, {**metrics, **compile_metrics}),
        target_dir=target_dir,
    )

    print("----------------- RUNTIME RESULTS -----------------")
    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize(rowwise=True)
    compare.print()

    if "pt2_" in args.funcs:
        print("----------------- COMPILE TIME RESULTS -----------------")
        print(compile_metrics)


if __name__ == "__main__":
    run(sys.argv[1:])
