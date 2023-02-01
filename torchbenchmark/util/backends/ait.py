
import os
import argparse
import torch
from torchbenchmark.util.backends import create_backend

from typing import List, Tuple
try:
    from fx2ait.acc_tracer import acc_tracer
    from fx2ait.ait_module import AITModule
    from fx2ait.fx2ait import AITInterpreter
except:
    # if fx2ait is not available, skip it.
    pass

def parse_ait_args(args: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda_graph", action='store_true', help="enable CUDA Graph")
    args, unknown_args = parser.parse_known_args(args)
    return args, unknown_args

@create_backend
def fx2ait(model: 'torchbenchmark.util.model.BenchmarkModel', backend_args: List[str]):
    AIT_WORK_PATH = os.path.join("/tmp", ".torchbench", "ait")
    assert model.dargs.precision == "fp16", f"AITemplate only support float16 precision, but get {model.dargs.precision}"
    OSS_AITModel = False
    try:
        # Load Non-OSS
        torch.ops.load_library("//deeplearning/ait:AITModel")
    except Exception:
        torch.ops.load_library("build/libait_model.so")
        OSS_AITModel = True

    ait_options, extra_args = parse_ait_args(backend_args)
    def _ait():
        mod, inputs = model.get_module()
        traced = acc_tracer.trace(mod, inputs)
        interp = AITInterpreter(traced, inputs, AIT_WORK_PATH, "logs")
        interp_result = interp.run()
        ctor = torch.classes.ait.AITModel if OSS_AITModel else torch.classes.fb.AITModel
        ait_mod = AITModule(
            ctor(
                interp_result.engine.lib_path,
                interp_result.input_names,
                interp_result.output_names,
                torch.float16,
                torch.float16,
                1,  # num_runtimes
            ),
        )

        ait_mod.engine.use_cuda_graph = ait_options.use_cuda_graph
    return _ait, extra_args
