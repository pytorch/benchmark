import itertools
import time
from datetime import datetime
from typing import List
import json
import numpy as np
import argparse
import re
import torch

from ..utils import REPO_PATH, add_path, get_output_dir, get_output_json, dump_output

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, load_model, TorchBenchModelConfig
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics, get_model_test_metrics
    import torchbenchmark.util.experiment.metrics

BM_NAME = "api-coverage"


def parse_func(func):
    if hasattr(func, '__module__'):
        module_name = func.__module__
        func_name = func.__name__
    else:
        if hasattr(func, '__qualname__'):
            func_name = func.__qualname__
            module_name = ''
        else:
            if type(func) == torch._C.Generator:
                func_name = 'torch._C.Generator'
                module_name = ''
            else:
                raise RuntimeError("no matched module and func name: ", func, type(func))
    return module_name, func_name


def generate_API_list():
    tmp_api_list = set()
    raw_all_apis = set(torch.overrides.get_testing_overrides().keys())
    # collect all items' attribute  `module` to a list
    for item in raw_all_apis:
        module_name, func_name = parse_func(item)
        # if (module_name, func_name) in api_list:
        # print("duplicated: ", (module_name, func_name))
        tmp_api_list.add((module_name, func_name))
    ignored_funcs = set([_ for _ in torch.overrides.get_ignored_functions() if _ not in [True, False]])
    tmp_ignored_api_list = set()
    for item in ignored_funcs:
        module_name, func_name = parse_func(item)
        tmp_ignored_api_list.add((module_name, func_name))
    return tmp_api_list, tmp_ignored_api_list

API_LIST, IGNORED_API_LIST = generate_API_list()


class CoverageMode(torch.overrides.TorchFunctionMode):

    def __init__(self, model='', output_file=None):
        self.model = model
        self.seen = set()
        self.api_used = set()
        self.output_file = output_file
        self.api_need_support = set()

    def check_func_in_APIs(self, func):
        module_name, func_name = parse_func(func)
        if (module_name, func_name) not in API_LIST:
            if (module_name, func_name) not in IGNORED_API_LIST and module_name != 'torch._ops.profiler':
                new_pair = (module_name, func_name)
                if new_pair not in self.api_need_support:
                    # debugging purpose
                    # print("not in API_LIST or IGNORED_API_LIST: (%s, %s)" % (module_name, func_name))
                    self.api_need_support.add((module_name, func_name))
        else:
            self.api_used.add((module_name, func_name))
            # debug
            # print("in APIs: ", (module_name, func_name))

    def get_api_coverage_rate(self):
        return len(self.api_used) / len(API_LIST)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.seen.add(func)
        if kwargs is None:
            kwargs = {}
        self.check_func_in_APIs(func)
        return func(*args, **kwargs)

    def commit(self):
        if self.output_file:
            with open(self.output_file, 'a') as f:
                for api in self.api_used:
                    f.write("%s,%s\n" % (api[0], api[1]))

    def update_api_used(self, output: set):
        for api in self.api_used:
            output.add(api)
    
    def update_need_support(self, output: set):
        for api in self.api_need_support:
            output.add(api)


def generate_model_config(model_name: str) -> List[TorchBenchModelConfig]:
    devices = ["cpu", "cuda"]
    tests = ["train", "eval"]
    cfgs = itertools.product(*[devices, tests])
    result = [TorchBenchModelConfig(
        name=model_name,
        device=device,
        test=test,
        batch_size=None,
        extra_args=[],
        extra_env=None,
    ) for device, test in cfgs]
    return result


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", default="",
                        help="Specify the models to run, default (empty) runs all models.")
    parser.add_argument("-d", "--device", default="cuda", help="Specify the device.")
    parser.add_argument("-t", "--test", default="eval,train", help="Specify the test.")
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
    torchbenchmark.util.experiment.metrics.BENCHMARK_ITERS = 1
    torchbenchmark.util.experiment.metrics.WARMUP_ROUNDS = 0
    single_round_result = []
    api_used = set()
    api_need_support = set()
    for cfg in filter(cfg_filter, cfgs):
        try:
            # load the model instance within the same process
            model = load_model(cfg)
            # get the model test metrics
            with CoverageMode('', '') as coverage:
                try:
                    get_model_test_metrics(model, metrics=["latencies"])
                finally:
                    coverage.update_api_used(api_used)
                    coverage.update_need_support(api_need_support)
        except NotImplementedError:
            # some models don't implement the test specified
            single_round_result.append({
                'cfg': cfg.__dict__,
                'raw_metrics': "NotImplemented",
            })
        except RuntimeError as e:
            single_round_result.append({
                'cfg': cfg.__dict__,
                'raw_metrics': f"RuntimeError: {e}",
            })

    # reduce full results to metrics
    # log detailed results in the .userbenchmark/model-stableness/logs/ directory
    log_dir = output_dir.joinpath("logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    fname = "logs-{}.json".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S"))
    full_fname = log_dir.joinpath(fname)
    with open(full_fname, 'w') as f:
        json.dump(single_round_result, f, indent=4)
    # log the api coverage
    api_coverage_fname = log_dir.joinpath("%s-api_coverage.csv" % fname)
    missed_apis = API_LIST - api_used
    with open(api_coverage_fname, 'w') as f:
        f.write("API coverage rate: %d/%d = %.2f%%\n" %
                (len(api_used), len(API_LIST), len(api_used) / len(API_LIST) * 100))
        f.write("=====Used APIs=====\n")
        f.write("module_name,func_name\n")
        for api in api_used:
            f.write("%s,%s\n" % (api[0], api[1]))
        f.write("=====Missed APIs=====\n")
        f.write("module_name,func_name\n")
        for api in missed_apis:
            f.write("%s,%s\n" % (api[0], api[1]))
    if api_need_support:
        api_need_support_fname = log_dir.joinpath("%s-api_need_support.csv" % fname)
        with open(api_need_support_fname, 'w') as f:
            f.write("APIs called but not in API_LIST and IGNORED_API_LIST\n")
            f.write("module_name,func_name\n")
            for api in api_need_support:
                f.write("%s,%s\n" % (api[0], api[1]))
    print("The detailed results are saved in %s" % api_coverage_fname)
