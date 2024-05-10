import yaml
import itertools
from typing import Any, Dict, List, Optional, Tuple
from torchbenchmark.util.experiment.instantiator import TorchBenchModelConfig, list_extended_models, get_model_set_from_model_name
from torchbenchmark.util.experiment.metrics import run_config

def _get_models(models: Optional[List[str]]=None, model_set: Optional[List[str]]=None) -> List[str]:
    result = set(models) if models else set()
    for s in model_set:
        result.union(set(list_extended_models(s)))
    return list(result) 

def config_obj_to_model_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, List[TorchBenchModelConfig]]]:
    models = _get_models(models=config.get("model", None), model_set=config.get("model_set", None))
    batch_sizes = config.get("batch_size", [None])
    tests = config.get("test", ["train", "eval"])
    devices = config.get("device", ["cuda"])
    precisions = config.get("precision", [None])
    metrics = config["metrics"]
    test_groups = config["test_groups"]
    result = {}
    for group_name in test_groups.keys():
        extra_args = test_groups[group_name].copy()
        cfgs = itertools.product(*[devices, tests, batch_sizes, precisions, models])
        for device, test, batch_size, precision, model_name in cfgs:
            if precision:
                extra_args = extra_args.extend(["--precision", precision])
            if batch_size:
                batch_size = int(batch_size)
            common_key = (device, test, batch_size, precision)
            if not common_key in result:
                result[common_key] = {}
            if not group_name in result[common_key]:
                result[common_key][group_name] = []
            result[common_key][group_name].append(
                TorchBenchModelConfig(
                    name=model_name,
                    device=device,
                    test=test,
                    batch_size=batch_size,
                    extra_args=extra_args,
                    extra_env=None,
                    metrics=metrics,
                )
            )
    return result


def _common_key_to_group_key(common_key: Tuple[str, str, int, str]):
    device, test, batch_size, precision = common_key
    return {
        "device": device,
        "test": test,
        "batch_size": batch_size,
        "precision": precision,
    }


def _config_result_to_group_result(group_name: str, model_name: str, metrics: Dict[str, Any], required_metrics: List[str]):
    # output metric format: <model_set>_<model_name>[<group_name>]_<metric_name>
    model_set = get_model_set_from_model_name(model_name)
    result = {}
    for metric in required_metrics:
        metric_name = f"{model_set}_{model_name}[{group_name}]_{metric}"
        result[metric_name] = metrics[metric]
    return result


def run_benchmark_group_config(group_config_file: str, dryrun: bool=False) -> List[Dict[str, Any]]:
    result = []
    with open(group_config_file, "r") as fp:
        config_obj = yaml.safe_load(fp)
    configs: Dict[str, Dict[str, List[TorchBenchModelConfig]]] = config_obj_to_model_configs(config_obj)
    for common_key in configs.keys():
        group_key = _common_key_to_group_key(common_key)
        group_result = {"group_key": group_key, "group_results": []}
        for group_name in configs[common_key]:
            group_result["group_results"] = [
                _config_result_to_group_result(
                    group_name=group_name,
                    model_name=x.name,
                    metrics=run_config(x, as_dict=True, dryrun=dryrun),
                    required_metrics=x.metrics)
                    for x in configs[common_key][group_name]
            ]
        result.append(group_result)
    return result
