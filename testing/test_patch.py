from typing import List

import torch
import torch.fx as fx
from functorch import (combine_state_for_ensemble,
                       make_functional_with_buffers, vmap)
from functorch._src.named_members_polyfill import (_named_buffers,
                                                   _named_parameters)
from functorch.compile import aot_function, aot_module, config, print_compile
from torch._subclasses import FakeTensor, FakeTensorMode

config.use_fake_tensor = True
from torchbenchmark import load_model_by_name
from torchbenchmark.models.nvidia_deeprecommender.reco_encoder.model.model import MSEloss

def fw_fake_compiler(fx_g: fx.GraphModule, inps, num_outs):
    # print(fx_g.code)
    output_node = [node for node in fx_g.graph.nodes if node.op == 'output'][0]
    output_data = [node.meta['val'] if node is not None else node for node in output_node.args[0]]
    def new_f(*args):
        return output_data
    return new_f

def bw_fake_compiler(fx_g: fx.GraphModule, inps):
    print(fx_g.code)
    output_node = [node for node in fx_g.graph.nodes if node.op == 'output'][0]
    output_data = [node.meta['val'] if node is not None else node for node in output_node.args[0]]
    def new_f(*args):
        return output_data
    return new_f



Model = load_model_by_name("nvidia_deeprecommender")


num_models = 2
batch_size = 128

model_list = [
    Model(device=torch.cuda.current_device(), test="train", batch_size = batch_size) for _ in range(num_models)
]


b_models = [model_list[i].model.rencoder for i in range(num_models)]

# inp = model_list[0].example_inputs
# targets = model_list[0].targets
# loss_fn = model_list[0].loss_fn
# sample_model = model_list[0]
# loss_fn = sample_model.cfg.loss
# input_batch = sample_model.example_inputs
# _gen_target = sample_model._gen_target
in_dims = (0, 0, None)
for model in b_models:
    model.train()

inp = model_list[0].model.toyinputs
loss_fn = MSEloss



func_model, params, buffers = combine_state_for_ensemble(b_models)
for p in params:
    p.requires_grad = True

b_models.clear()
model_list.clear()

# def compute_loss_dlrm(params, buffers, batch, targets):
#     gen = func_model(params, buffers, *batch)
#     loss = loss_fn(gen, targets)
#     return loss


# def compute_loss_mobilenet(params, buffers, batch, targets):
#     outputs = func_model(params, buffers, *batch)
#     loss = loss_fn(outputs, targets)
#     return loss


# def compute_loss(weights, buffers, batch):
#     out = func_model(weights, buffers, **batch)
#     return out.loss

def compute_loss_nvr(params, buffers, batch):
    outputs = func_model(params, buffers, batch)
    loss, num_ratings = loss_fn(outputs, batch)
    loss = loss / num_ratings
    return loss



# def compute_loss(params, buffers, input_batch):
#     output = func_model(params, buffers, input_batch)
#     target = _gen_target(output.shape[0])
#     loss = loss_fn(output, target)
#     return loss

parallel_func = vmap(compute_loss_nvr, in_dims=in_dims, randomness="same")

# out = parallel_func(params, buffers, inp, targets)
print(torch.cuda.memory_allocated())
aot_func = aot_function(parallel_func, fw_fake_compiler, bw_fake_compiler)
out = aot_func(params, buffers, inp)
print(out.size())
print(type(out))
print(out)
print(out.device)
out.sum().backward()
del params
del buffers
del inp
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
