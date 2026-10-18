import torch
import importlib 
import sys
import pprint
from pathlib import Path
import torch.utils._pytree as pytree

# Makes sure we setup torchbenchmark 
repo = Path(__file__).parent.parent.parent
sys.path.append(str(repo))

from userbenchmark.utils import dump_output
from userbenchmark.export_new_models import BM_NAME

models = [
    "hf_Qwen2",
    #"hf_simplescaling",
    "hf_minicpm",
    "kokoro",
]


def assert_equal(a, b):
    if a != b:
        raise AssertionError("not equal")


def compare_output(eager, export):
    flat_orig_outputs = pytree.tree_leaves(eager)
    flat_loaded_outputs = pytree.tree_leaves(export)

    for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
        assert_equal(type(orig), type(loaded))

        # torch.allclose doesn't work for float8
        if isinstance(orig, torch.Tensor) and orig.dtype not in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            if orig.is_meta:
                assert_equal(orig, loaded)
            else:
                if not torch.allclose(orig, loaded):
                    raise AssertionError("not equal")
        else:
            assert_equal(orig, loaded)
            

def get_model(name):
    model_module_ = importlib.import_module(f"torchbenchmark.models.{name}")
    model_cls = getattr(model_module_, "Model")
    model = model_cls(device="cuda", test="eval")
    return model

def run():
    metrics = {}
    errors = {}
    count_success = 0
    for model_name in models:
        print(f"Testing {model_name}")
        model = get_model(model_name)
        model, example_inputs = model.get_module()
        try:
            with torch.inference_mode():
                ep = torch.export.export(model, example_inputs[0], example_inputs[1], strict=False).module()
        except Exception as e:
            errors[model_name] = str(e)
            continue 
            
        try:
            with torch.inference_mode():
                compare_output(model(*example_inputs[0], **example_inputs[1]), ep.module()(*example_inputs[0], **example_inputs[1]))
        except Exception as e:
            errors[model_name] = str(e)
            continue
        count_success += 1
    
    metrics["success_rate"] = count_success / len(models)
    metrics["errors"] = errors
    
    result = {
        "name": BM_NAME,
        "environ": {
            "pytorch_git_version": torch.version.git_version,
        },
        "metrics": metrics,
    }
    pprint.pprint(result)
    dump_output(BM_NAME, result)

if __name__ == "__main__":
    run()
