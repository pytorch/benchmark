import torch
import importlib 
import sys
import pprint
#import torchbenchmark
from pathlib import Path

# Makes sure we setup torchbenchmark 
repo = Path(__file__).parent.parent.parent
sys.path.append(str(repo))

from userbenchmark.utils import dump_output
from userbenchmark.export_new_models import BM_NAME

models = [
    "hf_Qwen2"
]

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
        model = get_model(model_name)
        model, example_inputs = model.get_module()
        try:
            ep = torch.export.export(model, (), example_inputs, strict=False).module()
        except Exception as e:
            errors[model_name] = str(e)
            continue
        else:
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
