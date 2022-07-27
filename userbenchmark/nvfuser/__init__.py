import torch

import argparse
import json
import os
import time
import torch.utils.jit.log_extract as log_extract
from datetime import datetime
from typing import Any, List

def parse_fusers(extra_args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fusers",
        nargs="*",
        default=[],
        choices=["no_fuser", "fuser0", "fuser1", "fuser2"],
        help="List of fusers to run tests on")
    parser.add_argument("--filters", nargs="*", default=[], help='List of fuser microbenchmarks to test')
    parser.add_argument("--output", help="specifiy the output file name")
    args = parser.parse_args(extra_args)
    return args


class NVFuserBenchmark():
    def __init__(self, name, ir, warmup_runs=10, test_runs=20):
        self.name = name
        self.ir = ir
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs

    def run_test(self, inputs, fuser_name: str) -> float:
        if fuser_name == "no_fuser":
            return log_extract.run_baseline_no_fusion(self.ir, inputs)
        elif fuser_name == "nnc-static":
            return log_extract.run_nnc(self.ir, inputs, dynamic=False)
        elif fuser_name == "nnc-dynamic" or fuser_name == "fuser1":
            return log_extract.run_nnc(self.ir, inputs, dynamic=True)
        elif fuser_name == "fuser2" or fuser_name == "nvfuser":
            return log_extract.run_nvfuser(self.ir, inputs)
        assert False

    def get_inputs(self) -> List[Any]:
        _, inputs = log_extract.load_graph_and_inputs(self.ir)
        return inputs


def dump_metrics(metrics, output_name):
    output = {
        "name": "nvfuser",
        "environ": {"pytorch_git_version": torch.version.git_version},
        "metrics": metrics,
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.normpath(os.path.join(current_dir, "../../.userbenchmark/nvfuser/"))
    os.makedirs(target_dir, exist_ok=True)
    fname = "metrics-{}.json".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S"))
    full_fname = os.path.join(target_dir, fname)
    if output_name is not None:
        full_fname = output_name
    with open(full_fname, 'w') as f:
        json.dump(output, f, indent=4)


def run_nvfuser_microbenchmarks(extra_args: List[str]):
    from userbenchmark.nvfuser.ir import ir_list
    benchmarks = [NVFuserBenchmark(name, ir) for name, ir in ir_list]
    args = parse_fusers(extra_args)
    filters, fusers = args.filters, args.fusers
    if len(filters) > 0:
        benchmarks = [x for x in benchmarks if x.name in filters]
    if len(fusers) == 0:
        fusers = ["no_fuser", "nnc-static", "nnc-dynamic", "nvfuser"]

    metrics = {}
    for b in benchmarks:
        outputs = []
        for fuser in fusers:
            inputs = b.get_inputs()
            runtime = b.run_test(inputs, fuser)
            outputs.append((fuser, runtime))
            metrics[f"{fuser}:{b.name}"] = runtime
        print(f"{b.name}:", "; ".join(f"{name} = {time:.3f} ms" for name, time in outputs))
    dump_metrics(metrics, args.output)


def run(args: List[str]):
    run_nvfuser_microbenchmarks(extra_args=args)
