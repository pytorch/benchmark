import torch

def jit_model(model, eval_model, jit, optimize_for_inference=True):
    if jit:
        model = torch.jit.script(model)
        eval_model = torch.jit.script(eval_model)
        if optimize_for_inference:
            eval_model = torch.jit.optimize_for_inference(eval_model)
        assert isinstance(model, torch.jit.ScriptModule)
        assert isinstance(eval_model, torch.jit.ScriptModule)
    return model, eval_model