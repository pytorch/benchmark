import itertools
import time
from datetime import datetime
from typing import List
import yaml
import json
import numpy as np
import argparse

from ..utils import REPO_PATH, add_path, get_output_dir, get_output_json, dump_output

with add_path(REPO_PATH):
    from components._impl.workers.subprocess_rpc import UnserializableException, ChildTraceException
    from torchbenchmark.util.experiment.instantiator import list_models, load_model_isolated, TorchBenchModelConfig
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics

BM_NAME = "model-stableness"
# By default, use 7 percent as the threshold for stableness checking
STABLE_THRESHOLD = 0.07
# By default, run 15 iterations
DEFAULT_ITERATIONS = 15

def generate_model_config(model_name: str) -> List[TorchBenchModelConfig]:
    devices = ["cpu", "cuda"]
    tests = ["train", "eval"]
    cfgs = itertools.product(*[devices, tests])
    result = [TorchBenchModelConfig(
        name=model_name,
        device=device,
        test=test,
        batch_size=None,
        jit=False,
        extra_args=[],
        extra_env=None,
    ) for device, test in cfgs]
    return result

def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rounds", default=DEFAULT_ITERATIONS, type=int, help="Number of rounds to run to simulate measuring max delta in workflow.")
    parser.add_argument("-m", "--models", default="", help="Specify the models to run, default (empty) runs all models.")
    parser.add_argument("-d", "--device", default="cpu", help="Specify the device.")
    parser.add_argument("-t", "--test", default="eval", help="Specify the test.")
    parser.add_argument("-o", "--output", type=str, help="The default output json file.")
    args = parser.parse_args(args)
    return args

def _get_median_latencies(raw_metrics):
    has_all_latencies = len(list(filter(lambda x: 'latencies' in x, raw_metrics)))
    if not has_all_latencies == len(raw_metrics):
        return None
    median_latencies = list(map(lambda x: np.median(x['latencies']), raw_metrics))
    return median_latencies

def reduce_results(full_results):
    ub_metrics = {}
    latencies_by_cfg = {}
    for round_result in full_results:
        for result in round_result:
            cfg = result['cfg']
            cfg_name = f"{cfg['name']}_{cfg['device']}_{cfg['test']}_ootb_latencies"
            if not cfg_name in latencies_by_cfg:
                latencies_by_cfg[cfg_name] = []
            latencies_by_cfg[cfg_name].append(result['raw_metrics'])
    for cfg_name in latencies_by_cfg:
        raw_metrics = latencies_by_cfg[cfg_name]
        latencies = _get_median_latencies(raw_metrics)
        if latencies:
            ub_metrics[f"{cfg_name}_maxdelta"] = (max(latencies) - min(latencies)) / min(latencies)
        else:
            ub_metrics[f"{cfg_name}_maxdelta"] = -1.0
    return ub_metrics

def reduce_results_by_device(full_results):
    def _cfg_to_key(cfg):
        key = {}
        key["model"] = cfg["name"]
        key["test"] = cfg["test"]
        return frozenset(key.items())
    result_by_device = {}
    result_yaml_obj = {}
    for round_result in full_results:
        for result in round_result:
            cfg = result['cfg']
            device = cfg['device']
            raw_metrics = result['raw_metrics']
            result_by_device[device] = {} if not device in result_by_device else result_by_device[device]
            key = _cfg_to_key(cfg)
            result_by_device[device][key] = [] if not key in result_by_device[device] else result_by_device[device][key]
            result_by_device[device][key].append(raw_metrics)

    for device in result_by_device:
        result_yaml_obj[device] = []
        for key in result_by_device[device]:
            latencies = _get_median_latencies(result_by_device[device][key])
            if not latencies:
                continue
            max_delta = (max(latencies) - min(latencies)) / min(latencies)
            stable_obj = dict(key)
            stable_obj["max_delta"] = str(max_delta)
            if max_delta < STABLE_THRESHOLD:
                stable_obj["stable"] = True
            else:
                stable_obj["stable"] = False
            result_yaml_obj[device].append(stable_obj)
    return result_yaml_obj

def generate_filter(args: argparse.Namespace):
    allowed_models = args.models
    if allowed_models:
        allowed_models = allowed_models.split(",") if "," in allowed_models else [allowed_models]
    allowed_devices = args.device
    allowed_devices = allowed_devices.split(",") if "," in allowed_devices else [allowed_devices]
    allowed_tests = args.test
    allowed_tests = allowed_tests.split(",") if "," in allowed_tests else [allowed_tests]
    def cfg_filter(cfg: TorchBenchModelConfig) -> bool:
        if cfg.device in allowed_devices and cfg.test in allowed_tests:
            if not allowed_models:
                return True
            else:
                return cfg.name in allowed_models
        return False
    return cfg_filter

def run(args: List[str]):
    args = parse_args(args)
    output_dir = get_output_dir(BM_NAME)
    models = list_models()
    cfgs = list(itertools.chain(*map(generate_model_config, models)))
    cfg_filter = generate_filter(args)
    # run a model cfg and get latencies
    full_results = []
    for _round in range(args.rounds):
        single_round_result = []
        for cfg in filter(cfg_filter, cfgs):
            print(f"[Round {_round}/{args.rounds}] Running {cfg}")
            try:
                task = load_model_isolated(cfg)
                # get the model test metrics
                metrics: TorchBenchModelMetrics = get_model_test_metrics(task, metrics=["latencies"])
                single_round_result.append({
                    'cfg': cfg.__dict__,
                    'raw_metrics': metrics.__dict__,
                })
            except NotImplementedError:
                # some models don't implement the test specified
                single_round_result.append({
                    'cfg': cfg.__dict__,
                    'raw_metrics': "NotImplemented",
                })
            except ChildTraceException as exception:
                single_round_result.append({
                        'cfg': cfg.__dict__,
                        'raw_metrics': str(exception),
                })
            except UnserializableException as exception:
                single_round_result.append({
                        'cfg': cfg.__dict__,
                        'raw_metrics': exception.args_repr,
                })
            finally:
                # Remove task reference to trigger deletion in gc
                task = None
        full_results.append(single_round_result)
    print(full_results)
    ub_metrics = reduce_results(full_results)

    # reduce full results to metrics
    # log detailed results in the .userbenchmark/model-stableness/logs/ directory
    output_json = get_output_json(BM_NAME, ub_metrics)
    log_dir = output_dir.joinpath("logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    fname = "logs-{}.json".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S"))
    full_fname = log_dir.joinpath(fname)
    with open(full_fname, 'w') as f:
        json.dump(full_results, f, indent=4)
    # output userbenchmark metrics in the .userbenchmark/model-stableness directory
    print(output_json)
    dump_output(BM_NAME, output_json)
    # output the stableness result yaml
    yaml_dicts = reduce_results_by_device(full_results)
    for device in yaml_dicts:
        fname = f"summary-{device}.yaml"
        full_fname = log_dir.joinpath(fname)
        with open(full_fname, "w") as f:
            f.write(yaml.safe_dump(yaml_dicts[device]))
