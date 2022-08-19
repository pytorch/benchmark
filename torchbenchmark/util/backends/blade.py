import argparse
from charset_normalizer import logging
import torchdynamo
from torchdynamo.optimizations.backends import create_backend as dynamo_backend
from torchbenchmark.util.backends import create_backend
import torch
import torch_blade
from torch_blade import optimize as blade_optimize
from torch_blade import mlir, tensorrt
from typing import List


def parse_blade_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # enable ofi by default
    parser.add_argument("--trt", action='store_true', help="use blade trt backend")
    args = parser.parse_args(args)
    return args

@dynamo_backend
def blade_optimize_dynamo(subgraph, enable_fp16=False, use_trt=False):
    torch_config = torch_blade.config.Config()
    torch_config.enable_fp16 = enable_fp16
    if use_trt:
        torch_config.optimization_pipeline = torch_blade.tensorrt.backend_name()
    with torch_config, torch.no_grad():
        optimized_model = blade_optimize(
            subgraph.model.eval(),
            allow_tracing=True,
            model_inputs=tuple(subgraph.example_inputs),
        )

    if use_trt:
        num_engines = tensorrt.num_engines
        num_compiled_nodes = tensorrt.num_compiled_nodes
    else:
        num_engines = mlir.num_engines
        num_compiled_nodes = mlir.num_compiled_nodes
    
    if num_engines(optimized_model) == 0:
        logging.warning("blade none fusion group")
    torchdynamo.utils.counters["blade"]["clusters"] += num_engines(optimized_model)
    torchdynamo.utils.counters["blade"]["compiled_nodes"] += sum(num_compiled_nodes(optimized_model))

    # with open(f'model.code.py', 'a') as writer:
    #     writer.write(str(optimized_model.code))
    # with open(f'model.graph.txt', 'a') as writer:
    #     writer.write(str(optimized_model.graph))
    return optimized_model

@create_backend                   
def blade(model: 'torchbenchmark.util.model.BenchmarkModel', backend_args: List[str]):
    args = parse_blade_args(backend_args)
    module, example_inputs = model.get_module()
                                                          
    torch_config = torch_blade.config.Config()
    torch_config.enable_fp16 = model.dargs.precision == "fp16"
    if args.trt:
        torch_config.optimization_pipeline = torch_blade.tensorrt.backend_name()
    with torch_config, torch.no_grad():
        optimized_model = blade_optimize(
            module.eval(),
            allow_tracing=True,
            model_inputs=tuple(example_inputs),
        )
    
    model.set_module(optimized_model)

