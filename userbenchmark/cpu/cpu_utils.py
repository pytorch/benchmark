"""
Run PyTorch cpu benchmarking.
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_PATH = Path(__file__).absolute().parent.parent.parent
USERBENCHMARK_OUTPUT_PREFIX = ".userbenchmark"


class add_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


def list_metrics() -> List[str]:
    return ["latencies", "throughputs", "cpu_peak_mem"]


def parse_str_to_list(candidates):
    if isinstance(candidates, list):
        return candidates
    candidates = list(map(lambda x: x.strip(), candidates.split(",")))
    return candidates


def validate(candidates, choices: List[str]):
    """Validate the candidates provided by the user is valid"""
    if isinstance(candidates, List):
        for candidate in candidates:
            assert (
                candidate in choices
            ), f"Specified {candidate}, but not in available list: {choices}."
    else:
        assert (
            candidates in choices
        ), f"Specified {candidates}, but not in available list: {choices}."
    return candidates


def get_output_dir(bm_name, test_date=None):
    current_dir = Path(__file__).parent.absolute()
    bm_out_dir = current_dir.parent.parent.joinpath(
        USERBENCHMARK_OUTPUT_PREFIX, bm_name
    )
    test_date = (
        test_date
        if test_date
        else datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    )
    output_dir = bm_out_dir.joinpath("cpu-" + test_date)
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def get_output_json(bm_name, metrics):
    import torch

    return {
        "name": bm_name,
        "environ": {"pytorch_git_version": torch.version.git_version},
        "metrics": metrics,
    }


def dump_output(bm_name, output, output_dir=None, fname=None):
    output_dir = output_dir if output_dir else get_output_dir(bm_name)
    fname = fname if fname else "metrics-{}.json".format(os.getpid())
    full_fname = os.path.join(output_dir, fname)
    with open(full_fname, "w") as f:
        json.dump(output, f, indent=4)


def get_run(test_dir: Path):
    run = {}
    testdir_name = test_dir.name
    regex = "(.*)-(.*)"
    g = re.match(regex, testdir_name).groups()
    run["model"] = g[0]
    run["test"] = g[1]
    run["results"] = []
    ins_jsons = filter(lambda x: x.is_file(), test_dir.iterdir())
    for ins_json in ins_jsons:
        with open(ins_json, "r") as ij:
            run["results"].append(json.load(ij))
    return run


def get_runs(work_dir: Path):
    runs = []
    for subdir in filter(lambda x: x.is_dir(), work_dir.iterdir()):
        run = get_run(subdir)
        runs.append(run)
    return runs


def add_test_results(runs, result_metrics):
    # metrics name examples:
    # timm_regnet-eval_latency
    # timm_regnet-eval_cmem
    for run in runs:
        run_base_name = f"{run['model']}-{run['test']}"
        ins_number = len(run["results"])
        assert ins_number
        latency_metric = "latency" in run["results"][0]["metrics"]
        iter_latencies_metric = "iter_latencies" in run["results"][0]["metrics"]
        throughput_metric = "throughput" in run["results"][0]["metrics"]
        iter_throughputs_metric = "iter_throughputs" in run["results"][0]["metrics"]
        cmem_metric = "cpu_peak_mem" in run["results"][0]["metrics"]
        latency_sum = 0
        iter_latencies = []
        throughput_sum = 0
        iter_throughputs = []
        cmem_sum = 0
        for ins_res in run["results"]:
            if latency_metric:
                latency_sum += ins_res["metrics"]["latency"]
            if iter_latencies_metric:
                iter_latencies += ins_res["metrics"]["iter_latencies"]
            if throughput_metric:
                throughput_sum += ins_res["metrics"]["throughput"]
            if iter_throughputs_metric:
                iter_throughputs += ins_res["metrics"]["iter_throughputs"]
            if cmem_metric:
                cmem_sum += ins_res["metrics"]["cpu_peak_mem"]
        if latency_metric:
            result_metrics[f"{run_base_name}_latency"] = latency_sum / ins_number
        if iter_latencies_metric:
            result_metrics[f"{run_base_name}_iter_latencies"] = iter_latencies
        if throughput_metric:
            result_metrics[f"{run_base_name}_throughput"] = throughput_sum
        if iter_throughputs_metric:
            result_metrics[f"{run_base_name}_iter_throughputs"] = iter_throughputs
        if cmem_metric:
            result_metrics[f"{run_base_name}_cmem"] = cmem_sum / ins_number
    return result_metrics


def analyze(result_dir):
    result_dir = Path(result_dir)
    assert result_dir.is_dir(), f"Expected directory {str(result_dir)} doesn't exist."
    result_metrics = {}
    runs = get_runs(result_dir)
    cpu_train = list(filter(lambda x: x["test"] == "train", runs))
    if len(cpu_train):
        add_test_results(cpu_train, result_metrics)
    cpu_eval = list(filter(lambda x: x["test"] == "eval", runs))
    if len(cpu_eval):
        add_test_results(cpu_eval, result_metrics)
    return result_metrics
