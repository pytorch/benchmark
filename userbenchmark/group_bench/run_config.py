import yaml
from typing import Any, Dict, List, Optional
from torchbenchmark.util.experiment.instantiator import TorchBenchModelConfig, list_extended_models
from torchbenchmark.util.experiment.metrics import run_config

def _get_models(models: Optional[List[str]]=None, model_set: Optional[List[str]]=None) -> List[str]:
    result = set(models) if models else set()
    for s in model_set:
        result.add(list_extended_models(s))
    return list(result) 

def config_obj_to_model_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, TorchBenchModelConfig]]:
    models = _get_models(config.get("model"), config.get("model_set"))
    tests = config["test"]
    devices = config["device"]
    metrics = config["metrics"]


def run_benchmark_group_config(group_config_file: str, dryrun: bool=False) -> Dict[str, Dict[str, Any]]:
    result = {}
    with open(group_config_file, "r") as fp:
        config_obj = yaml.safe_load(fp)
    configs = config_obj_to_model_configs(config_obj)
    for key in configs.keys():
        benchmark_results = [(key, run_config(configs[key][x], as_dict=True, dryrun=dryrun)) for x in configs[key].keys()]
        result[key] = dict(benchmark_results)
    return result
