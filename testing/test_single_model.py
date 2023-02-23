from typing import List

import torch
import torch.fx as fx
from functorch import (combine_state_for_ensemble,
                       make_functional_with_buffers, vmap)
from functorch._src.named_members_polyfill import (_named_buffers,
                                                   _named_parameters)
from functorch.compile import aot_function, aot_module, config, print_compile, make_boxed_func
from torch._subclasses import FakeTensor, FakeTensorMode

config.use_fake_tensor = True
import sys
from torchbenchmark import load_model_by_name


def fake_compiler(fx_g: fx.GraphModule, inps):
    # print(fx_g.code)
    output_node = [node for node in fx_g.graph.nodes if node.op == 'output'][0]
    output_data = [node.meta['val'] if node is not None else node for node in output_node.args[0]]
    def new_f(args):
        return output_data
    new_f._boxed_call = True
    return new_f



Model = load_model_by_name("hf_Bert")
batch_size = 16

model = Model(device="cuda", test="train", batch_size=batch_size) 



inp = model.example_inputs
# loss_fn = model.loss_fn
# targets = model.targets

func_model, params, buffers = make_functional_with_buffers(model.model, disable_autograd_tracking=True)
for p in params:
    p.requires_grad = True


def compute_loss(params, buffers, batch):
    out = func_model(params, buffers, **batch)
    return out.loss

# def compute_loss_dlrm(params, buffers, batch, targets):
#     gen = func_model(params, buffers, *batch)
#     loss = loss_fn(gen, targets)
#     return loss

aot_func = aot_function(compute_loss, fake_compiler)
out = aot_func(params, buffers, inp)
print(out.size())
print(type(out))
print(out)
print(out.device)
out.backward()

print(torch.cuda.memory_allocated())




# g = {}
# def fake_wrapper(gtype):
# def fake_compiler(fx_g, inps):
# print(fx_g.code)
# nonlocal gtype
# g[gtype] = fx_g
# return fx_g
# return fake_compiler
# aot_func = aot_function(parallel_func, fake_wrapper("forward"), fake_wrapper("backward"))
