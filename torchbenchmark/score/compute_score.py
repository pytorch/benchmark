
"""
Compute the benchmark score given a frozen score configuration and current benchmark data.
"""
import argparse
import json
import math
import sys
import os
import yaml
import importlib

from tabulate import tabulate
from pathlib import Path
from collections import defaultdict

SPEC_FILE_DEFAULT = "score.yml"
TARGET_SCORE_DEFAULT = 1000

def get_benchmark_norms(data):
    """
    Helper function which gets the normalization values per benchmark
    by going through the reference data file.

    This reference data can be user provided or by default the first file
    in the directory of data files is considered as the reference data file
    """
    benchmark_norms = {b['name']: b['stats']['mean'] for b in data['benchmarks']}
    return benchmark_norms

def get_weights_from_spec(spec):
    """
    Calculates the static benchmark weights by iterating the spec
    file and constructs a b_weight dictionary with (key, value) pair
    is (task, weight_for_benchmark_per_task)

    The spec here can be user defined .yaml file or if no spec is defined,
    the default score.yaml is used
    """
    b_weight = defaultdict(float)
    category_spec = spec['hierarchy']['model']
    domain_weight = 1.0/ len(category_spec)
    for domain in category_spec:
        tasks = category_spec[domain]
        task_weight = 1.0 / len(tasks)
        for task in tasks:
            benchmarks = tasks[task]
            benchmark_weight = 1.0 / len(benchmarks)
            b_weight[task] = domain_weight * task_weight * benchmark_weight

    return b_weight


def get_benchmark_configuration(b):
    """
    This helper function extracts the configuration details
    from the benchmark name.
    For eg., if benchmark_name = test_eval[pytorch_struct-cuda-jit]
            test, device, mode = eval, cuda, jit
            model_name = pytorch_struct

    This helper also extracts the task and model_name of the benchmark
    The function returns a tuple of the configs, model task, model_name
    """
    test, device, mode = "train", "cuda", "eager"
    if "eval" in b:
        test = "eval"
    if "cpu" in b:
        device = "cpu"
    if "jit" in b:
        mode = "jit"

    # Extract the model name from the benchmark b
    pos1 = b.index("[")
    pos2 = b.index("-"+device)
    model_name = b[pos1+1:pos2]

    # Get the Model task value by reading the Model attributes.
    p = Path(__file__).parent.joinpath('../../models/'+ model_name)
    if(p):
        module = importlib.import_module(f'torchbenchmark.models.{model_name}', package=__name__)
        Model = getattr(module, 'Model')
    return (test, device, mode), Model.task.value, model_name

def construct_benchmark_run_db(data):
    """
    Construct a benchmark database by going over through the data file
    for the run and update the dictionary by task and model_name

    For eg., the (key, value) for this dictionary is of the form
    [generation][pytorch_stargan] = [(1.2, (eval, cuda, jit),
                                    test_eval[pytorch_stargan-cuda-jit])]
    """
    found_benchmarks = defaultdict(dict)

    for b in data['benchmarks']:
        name, mean = b['name'], b['stats']['mean']
        config, task, model_name = get_benchmark_configuration(name)
        if not found_benchmarks[task]:
            found_benchmarks[task] = defaultdict(list)
        # Append the tuple(mean, config, model_name) for all the configs the
        # benchmark was run with.
        found_benchmarks[task][model_name].append((mean, config, name))
    return found_benchmarks

def get_score_per_config(data, b_weight, norm):
    """
    This function iterates over found benchmark dictionary
    and calculates the weight_sum and benchmark_score.
    A score_db is then constructed to calculate the cummulative
    score per config. Here config refers to device, mode and test
    configurations the benchmark was run on.

    For eg., if the benchmark was run in eval mode on a GPU in Torchscript JIT,
                config = (train, cuda, jit)

    This helper returns the score_db .

    """
    found_benchmarks = construct_benchmark_run_db(data)
    weight_sum = 0.0
    score_db = defaultdict(float)

    for task, models in found_benchmarks.items():
        for name, all_configs in models.items():
            weight = b_weight[task] * (1.0/len(all_configs))
            for mean, config, benchmark in all_configs:
                weight_sum += weight
                benchmark_score = weight * math.log(norm[benchmark] / mean)
                score_db[config] += benchmark_score

    assert abs(weight_sum - 1.0) < 1e-6, f"Bad configuration, weights don't sum to 1, but {weight_sum}"
    return score_db

def compute_score(weight_db, data, target, norm):
    """
    This API calculates the total score for all the benchmarks
    that was run  by reading the data (.json) file.
    The weights are then calibrated to the target score.

    # TODO:
    # score_db: score_per_config for now is just calculated.
    # This needs to passed to plot_sweep.py to plot all the
    # scores per config.
    """
    score = 0.0
    score_db = get_score_per_config(data, weight_db, norm)

    for config, scores in score_db.items():
        score += scores
        score_db[config] = target * math.exp(scores * 0.125)

    score = target * math.exp(score)
    return score
