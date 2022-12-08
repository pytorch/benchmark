import itertools
import time
from datetime import datetime
from typing import List
import json
import argparse

from ..utils import REPO_PATH, add_path, get_output_dir, get_output_json, dump_output

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, load_model, TorchBenchModelConfig
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics

BM_NAME = "model-stableness"

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
    parser.add_argument("-m", "--models", default="", help="Specify the models to run, default (empty) runs all models.")
    parser.add_argument("-d", "--device", default="cpu", help="Specify the device.")
    parser.add_argument("-t", "--test", default="eval", help="Specify the test.")
    parser.add_argument("-o", "--output", type=str, help="The default output json file.")
    args = parser.parse_args(args)
    return args

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
    detailed_results = []
    ub_metrics = {}
    for cfg in filter(cfg_filter, cfgs):
        try:
            # load the model instance within the same process
            model = load_model(cfg)
            # get the model test metrics
            metrics: TorchBenchModelMetrics = get_model_test_metrics(model)
            latencies = metrics.latencies
            max_delta = (max(latencies) - min(latencies)) / min(latencies)
            detailed_results.append({
                'cfg': cfg.__dict__,
                'raw_metrics': metrics.__dict__,
                'max_delta': max_delta,
            })
            metric_name = f"{cfg.name}_{cfg.device}_{cfg.test}_ootb_maxdelta"
            ub_metrics[metric_name] = max_delta
        except NotImplementedError:
            # some models don't implement the test specified
            detailed_results.append({
                'cfg': cfg.__dict__,
                'raw_metrics': "NotImplemented",
            })
        except RuntimeError as e:
            detailed_results.append({
                'cfg': cfg.__dict__,
                'raw_metrics': f"RuntimeError: {e}",
            })

    print(detailed_results)
    # log detailed results in the .userbenchmark/model-stableness/logs/ directory
    output_json = get_output_json(BM_NAME, ub_metrics)
    log_dir = output_dir.joinpath("logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    fname = "logs-{}.json".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S"))
    full_fname = log_dir.joinpath(fname)
    with open(full_fname, 'w') as f:
        json.dump(detailed_results, f, indent=4)
    # output userbenchmark metrics in the .userbenchmark/model-stableness directory
    print(output_json)
    dump_output(output_json)
