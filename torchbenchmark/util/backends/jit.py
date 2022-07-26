import torch

from torchbenchmark.util.backends import create_backend

@create_backend
def torchscript(model: 'torchbenchmark.util.model.BenchmarkModel', **kwargs):
    # customized jit callback function
    model.jit = True
    if hasattr(model, 'jit_callback'):
        model.jit_callback()
        return
    module, example_inputs = model.get_module()
    if hasattr(torch.jit, '_script_pdt'):
        module = torch.jit._script_pdt(module, example_inputs=[example_inputs, ])
    else:
        module = torch.jit.script(module, example_inputs=[example_inputs, ])
    optimized_for_inference = False if "optimize_for_inference" in kwargs and \
                                    not kwargs["optimize_for_inference"] else True
    if model.test == "eval" and optimized_for_inference:
        module = torch.jit.optimize_for_inference(module)
    model.set_module(module)
