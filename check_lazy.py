"""
A 2-layer util for getting stats about lazy tensor models in torchbench.
(1) by default with no arguments, this triggers a sweep where one subprocess per model checks the model behavior
(2) with arguments specifying the model and mode, runs just that model in the current process and collects stats.

DANGER: make sure to `python install.py` for torchbench first or otherwise make sure the benchmark you are going to run
        has been installed.  This script intentionally does not automate or enforce setup steps.
"""
import argparse

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import torch
import importlib
from torchbenchmark import _list_model_paths
import lazy_tensor_core
import datetime
lazy_tensor_core._LAZYC._ltc_init_ts_backend()

# The following models don't have the corresponding tests.
skip_tests = { 'eval': {'pytorch_struct'},
               'train': {'pyhpc_equation_of_state', 'pyhpc_isoneutral_mixing'}}

def list_model_names():
    return [os.path.basename(model_path) for model_path in _list_model_paths()]

def run_model_command(name, test, output_file):
    script = os.path.abspath(__file__)
    command = [sys.executable, script,
               "--check_model", name,
               "--test", test,
               "--output_file", output_file]
    return command

def process_model_stats(name, test, model_output_file):
    try:
        model_stats = json.load(model_output_file)
        model_stats['model'] = name
        model_stats['test'] = test

    except json.decoder.JSONDecodeError:
        model_stats = {
            'model': name, 'test': test,
            'exception': 'failed to decode subprocess output as json (possible c++ assertion, TODO parse stderr)'
        }
    return model_stats

def sweep_models(output_filename, tests=['eval', 'train']):
    stats = []
    with open(output_filename, 'w') as output_file:
        for name in list_model_names():
            for test in tests:
                if name in skip_tests[test]:
                    continue

                with tempfile.NamedTemporaryFile(mode='r') as model_output_file:
                    env = os.environ
                    env["LTC_TS_CUDA"] = "1"
                    launch_command = run_model_command(name, test, model_output_file.name)
                    # python stdlib didn't include tzones until 3.9
                    PST_OFFSET = datetime.timedelta(hours=8)
                    dt = datetime.datetime.now() - PST_OFFSET
                    print(f"{dt} : Running launch_command {' '.join(launch_command)}")
                    try:
                        rc = subprocess.call(launch_command,
                                            env=env,
                                            timeout = 360, # 6 minutes, 2 min max per iter
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.STDOUT)
                        model_stats = process_model_stats(name, test, model_output_file)
                        print(f"Ran {name}:{test}, RC={rc}, stats: ")
                    except subprocess.TimeoutExpired:
                        model_stats = {
                            'model': name, 'test': test,
                            'exception': 'model timed out'
                        }
                        print(f"Ran {name}:{test}, RC=1, stats: ")

                    stats.append(model_stats)
                    print(model_stats)
                    json.dump(model_stats, output_file)

def json_to_csv(json_file, csv_file):
    d = json.JSONDecoder()

    stats = []
    with open(json_file,'r') as f:
        buf = f.read();
        while True:
            try:
                model_stats, pos = d.raw_decode(buf)
                stats.append(model_stats)
            except ValueError:
                break
            buf = buf[pos:]

    sorted_keys = ["model", "test", "times", "CachedCompile", "UncachedCompile", "aten_ops", "exception"]
    keys = set().union(*(d.keys() for d in stats)) - set(sorted_keys)
    keys = sorted_keys + list(keys)
    with open(csv_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(stats)

def get_model_class(model_name):
    try:
        module = importlib.import_module(f'.models.{model_name}', package="torchbenchmark")
        Model = getattr(module, 'Model', None)
        if Model is None:
             raise RuntimeError(f"{module} does not define attribute Model, skip it")
        if not hasattr(Model, 'name'):
            Model.name = model_name
        return Model
    except ModuleNotFoundError as e:
        raise RuntimeError(f"Could not find dependent module {e.name} for Model {model_name}, skip it")

def _check_model(model_name, test, output_file, niter):
    import lazy_tensor_core.core.lazy_model as ltm
    import lazy_tensor_core.debug.metrics as metrics
    torch.manual_seed(42)
    times = []
    for i in range(niter):
        t0 = time.time()
        test()
        ltm.mark_step()
        ltm.wait_device_ops()
        times.append(time.time() - t0)
    raw_counters = ["CachedCompile", "CreateLtcTensor", "DestroyLtcTensor", "DeviceDataCacheMiss", "MarkStep", "UncachedCompile"]
    aten_ops = [n[5:] for n in metrics.counter_names() if 'aten::' in n]
    lazy_ops = [n[5:] for n in metrics.counter_names() if 'lazy::' in n]
    stats = {
        "model":  model_name,
        "times": times,
        **{n: metrics.counter_value(n) for n in metrics.counter_names() if n in raw_counters},
        "aten_ops": aten_ops,
        "lazy_ops": lazy_ops,
    }

    return stats

def check_model(model_name, test, output_file, niter=1):
    try:
        stats = _check_model(model_name, test, output_file, niter)
    except Exception as e:
        stats = {"model": model_name, "exception": str(e)}
    with open(output_file, 'w') as f:
        json.dump(stats, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--json_to_csv", type=str, help="Convert partial json file (named via this arg) into csv file (--output_file).")
    parser.add_argument("--check_model", type=str, help="Check this particular model.")
    parser.add_argument("--output_file", required=True,  type=str, help="Write model output to this file (stdout by default)")
    parser.add_argument("--device", choices=["lazy"], default="lazy", help="Which mode to run.")
    parser.add_argument("--test", choices=["eval",  "train"], default="eval", help="Which test to run.")
    parser.add_argument("--list_models", action="store_true", help="List the available models and exit.")
    args = parser.parse_args()
    if args.list_models:
        print(list_model_names())
        exit(0)
    if args.json_to_csv:
        json_to_csv(args.json_to_csv, args.output_file)
        exit(0)

    if args.check_model:
        Model = get_model_class(args.check_model)
        model = Model(device=args.device, jit=False)
        test = getattr(model, args.test)
        exit(check_model(args.check_model, test, args.output_file))

    exit(sweep_models(args.output_file))

