from typing import Any, Dict, List, Tuple
from torchbenchmark import load_model_by_name
import torch
from userbenchmark.utils import dump_output, get_output_json


BM_NAME = "optim"

MODEL_NAMES = [
    "BERT_pytorch",
    # "hf_T5_large",
    "resnet18",
]

OPTIMIZERS = [
    (torch.optim.Adadelta, {"foreach": True,}),
    (torch.optim.Adagrad, {"foreach": True,}),
    (torch.optim.Adam, {"foreach": True, "fused": True}),
    (torch.optim.AdamW, {"foreach": True,}),
    # (torch.optim.SparseAdam, {}),
    (torch.optim.Adamax, {"foreach": True,}),
    (torch.optim.ASGD, {"foreach": True,}),
    # (torch.optim.SGD, {"foreach": True,}),
    (torch.optim.RAdam, {"foreach": True,}),
    (torch.optim.Rprop, {"foreach": True,}),
    (torch.optim.RMSprop, {"foreach": True,}),
    (torch.optim.NAdam, {"foreach": True,}),
    # (torch.optim.LBFGS, {}),
]

devices = ["cuda", "cpu"]

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
    default_m, default_inputs, default_params = get_model_deets(Model(device=device, test="train"))
    if foreach is None and fused is None:
        default_o = Optim(default_params)
    elif foreach is None and fused:
        default_o = Optim(default_params, fused=fused)
    elif fused is None and foreach:
        default_o = Optim(default_params, foreach=foreach)
    else:
        default_o = Optim(default_params, foreach=foreach, fused=fused)
    print('Running the model!', modelName, device, Optim, foreach, fused)
    forward_and_backward(default_m, default_inputs)
    print('Done the forward and backward')
    optimizer_step(default_o)
    print('Done the optimizer step')

def run_benchmarks() -> List[float]:
    for mn in MODEL_NAMES:
        for d in devices:
            for O, impls in OPTIMIZERS:
                run_model(mn, d, O, None, None)
                if 'foreach' in impls and impls["foreach"]:
                    run_model(mn, d, O, True, None)
                # fused requires params to be floats on CUDA
                if 'fused' in impls and impls["fused"] and d == 'cuda':
                    run_model(mn, d, O, None, True)


def run(args: List[str]):
    run_benchmarks()
    metrics: Dict[str, float] = {} 
    # gotta output a JSON now do I
    dump_output(BM_NAME, get_output_json(BM_NAME, metrics))


if __name__ == "__main__":
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