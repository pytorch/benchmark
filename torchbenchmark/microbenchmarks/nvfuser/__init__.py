from contextlib import contextmanager
from typing import Any, List, Tuple
from torch.testing import make_tensor
import argparse
import random
import torch
import time


# TODO - a lot of this was copied from pytorch/jit/scripts/log_extract.py,
# should we put it somewhere in torch? (and where?)

@contextmanager
def no_fuser(*args, **kwargs):
    old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
    old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
    old_texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
    old_nvfuser_state = torch._C._jit_nvfuser_enabled()

    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)

    try:
        yield
    finally:
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(old_gpu_fuse)
        torch._C._jit_set_texpr_fuser_enabled(old_texpr_fuser_state)
        torch._C._jit_set_nvfuser_enabled(old_nvfuser_state)


def make_tensor_from_type(inp_type: torch._C.TensorType):
    if inp_type.requires_grad() is not False:
        raise NotImplementedError("Tensors with requires_grad are not implemented")
    return make_tensor(
        inp_type.sizes(),
        dtype=inp_type.dtype(),
        device=inp_type.device())


def load_graph_and_inputs(ir: str) -> Tuple[Any, List[Any]]:
    graph = torch._C.parse_ir(ir)
    graph.makeMultiOutputIntoTuple()
    inputs = []
    for inp in graph.inputs():
        if isinstance(inp.type(), torch._C.FloatType):
            inputs.append(random.uniform(.1, 100))
        elif isinstance(inp.type(), torch._C.IntType):
            inputs.append(random.randint(1, 100))
        elif isinstance(inp.type(), torch._C.TensorType):
            inputs.append(make_tensor_from_type(inp.type()))
        else:
            raise NotImplementedError(f"A default value is not implemented for type {inp.type()}")

    func = torch._C._create_function_from_graph("forward", graph)
    torch._C._jit_pass_erase_shape_information(func.graph)
    return (func, inputs)


def time_cuda(fn, inputs, test_runs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    torch.cuda.synchronize()
    for i in range(test_runs):
        fn(*inputs)
        torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / test_runs


def time_cpu(fn, inputs, test_runs):
    s = time.perf_counter()
    for _ in range(test_runs):
        fn(*inputs)
    e = time.perf_counter()
    return (e - s) / test_runs


def run_test(ir, inputs, *, warmup_runs=10, test_runs=20) -> float:
    graph, _ = load_graph_and_inputs(ir)
    for _ in range(warmup_runs):
        graph(*inputs)

    is_cpu = None
    for input in inputs:
        if isinstance(input, torch.Tensor):
            is_cpu = input.device.type == "cpu"
            break
    assert is_cpu is not None

    out = time_cpu(graph, inputs, test_runs) if is_cpu else time_cuda(graph, inputs, test_runs)
    return out


def parse_fusers(extra_args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fusers",
        nargs="*",
        default=[],
        choices=["no_fuser", "fuser0", "fuser1", "fuser2"],
        help="List of fusers to run tests on")
    args = parser.parse_args(extra_args)
    return args.fusers


class NVFuserBenchmark():
    def __init__(self, name, ir, warmup_runs=10, test_runs=20):
        self.name = name
        self.ir = ir
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs

    def run_test(self, inputs, fuser_name: str) -> float:
        if fuser_name == "no_fuser":
            with no_fuser():
                return run_test(self.ir, inputs, warmup_runs=self.warmup_runs, test_runs=self.test_runs)
        with torch.jit.fuser(fuser_name):
            return run_test(self.ir, inputs, warmup_runs=self.warmup_runs, test_runs=self.test_runs)

    def get_inputs(self) -> List[Any]:
        _, inputs = load_graph_and_inputs(self.ir)
        return inputs


def run_nvfuser_microbenchmarks(filters: List[str], extra_args: List[str]):
    from torchbenchmark.microbenchmarks.nvfuser.ir import ir_list
    benchmarks = [NVFuserBenchmark(name, ir) for name, ir in ir_list]
    if len(filters) > 0:
        benchmarks = [x for x in benchmarks if x.name in filters]

    fusers = parse_fusers(extra_args)
    if len(fusers) == 0:
        fusers = ["no_fuser", "fuser1", "fuser2"]

    for b in benchmarks:
        outputs = []
        for fuser in fusers:
            inputs = b.get_inputs()
            outputs.append((fuser, b.run_test(inputs, fuser)))
        print(f"{b.name}:", "; ".join(f"{name} = {time:.3f} ms" for name, time in outputs))
