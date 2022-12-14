import os
import pathlib
import importlib
from dataclasses import dataclass
from typing import List, Dict, Any

E2E_MODEL_DIR = 'e2e_models'

def _list_model_paths() -> List[str]:
    p = pathlib.Path(__file__).parent.joinpath(E2E_MODEL_DIR)
    return sorted(str(child.absolute()) for child in p.iterdir() if child.is_dir())

@dataclass
class E2EBenchmarkResult:
    device: str
    device_num: int
    test: str
    num_examples: int
    num_epochs: int
    batch_size: int
    result: Dict[str, Any]

def load_e2e_model_by_name(model):
    models = filter(lambda x: model.lower() == x.lower(),
                    map(lambda y: os.path.basename(y), _list_model_paths()))
    models = list(models)
    if not models:
        return None
    assert len(models) == 1, f"Found more than one models {models} with the exact name: {model}"
    model_name = models[0]
    try:
        module = importlib.import_module(f'torchbenchmark.e2e_models.{model_name}', package=__name__)
    except ModuleNotFoundError as e:
        print(f"Warning: Could not find dependent module {e.name} for Model {model_name}, skip it: {e}")
        return None
    Model = getattr(module, 'Model', None)
    if Model is None:
        print(f"Warning: {module} does not define attribute Model, skip it")
        return None
    if not hasattr(Model, 'name'):
        Model.name = model_name
    return Model
