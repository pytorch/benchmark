
"""
Compute the benchmark score given a frozen score configuration and current benchmark data.
"""
import argparse
import json
import math
import sys
import os
import re
import yaml
import importlib

from tabulate import tabulate
from pathlib import Path
from collections import defaultdict


TARGET_SCORE_DEFAULT = 1000
SPEC_FILE_DEFAULT = Path(__file__).parent.joinpath("score.yml")
TORCHBENCH_V0_REF_DATA = Path(__file__).parent.joinpath("configs/v0/config-v0.yaml")
TORCHBENCH_V1_REF_DATA = "" # Placeholder for v1 score

def _get_model_task(model_name):
    """
    Helper function which extracts the task the model belongs to
    by iterating over the Model attributes.
    """
    MODEL_PATH="../models"
    p = Path(__file__).parent.joinpath(MODEL_PATH + model_name)
    if(p):
        module = importlib.import_module(f'torchbenchmark.models.{model_name}', package=__name__)
        Model = getattr(module, 'Model')
    return Model.task.value

class TorchBenchScore:
    def __init__(self, ref_data, spec=SPEC_FILE_DEFAULT, target=TARGET_SCORE_DEFAULT):
        self.spec = spec
        self.target = target
        self.ref_data = ref_data
        self.weights = None
        self.norm = None

        # Setup weights and benchmark norms
        self._setup_weights()
        self._setup_benchmark_norms()

    def _setup_weights(self):
        """
        Calculates the static benchmark weights by iterating the spec
        file and constructs a dictionary with (key, value) pair
        is (task, weight_for_benchmark_per_task)
        """
        # Load the spec file
        with open(self.spec) as spec_file:
            spec = yaml.full_load(spec_file)

        self.weights = defaultdict(float)
        category_spec = spec['hierarchy']['model']
        domain_weight = 1.0/ len(category_spec)
        for domain in category_spec:
            tasks = category_spec[domain]
            task_weight = 1.0 / len(tasks)
            for task in tasks:
                benchmarks = tasks[task]
                benchmark_weight = 1.0 / len(benchmarks)
                self.weights[task] = domain_weight * task_weight * benchmark_weight

    def _setup_benchmark_norms(self):
        """
        Helper function which gets the normalization values per benchmark
        by going through the reference data file.
        """
        if self.ref_data in [TORCHBENCH_V0_REF_DATA, TORCHBENCH_V1_REF_DATA]:
            with open(self.ref_data) as ref_file:
                ref = yaml.full_load(ref_file)
            self.norm = {b: ref['benchmarks'][b]['norm'] for b in ref['benchmarks']}
        else:
            self.norm = {b['name']: b['stats']['mean'] for b in self.ref_data['benchmarks']}

    def get_score_per_config(self, data):
        """
        This function iterates over found benchmark dictionary
        and calculates the weight_sum and benchmark_score.
        A score_db is then constructed to calculate the cumulative
        score per config. Here config refers to device, mode and test
        configurations the benchmark was run on.

        For eg., if the benchmark was run in eval mode on a GPU in Torchscript JIT,
                    config = (train, cuda, jit)

        This helper returns the score_db .

        """
        found_benchmarks = defaultdict(lambda: defaultdict(list))
        score_db = defaultdict(float)

        # Construct a benchmark database by going over through the data file
        # for the run and update the dictionary by task and model_name
        for b in data['benchmarks']:
            name, mean = b['name'], b['stats']['mean']
            test, model_name, device, mode = re.match(r"test_(.*)\[(.*)\-(.*)\-(.*)\]", name).groups()
            config = (test, device, mode)
            task = _get_model_task(model_name)
            found_benchmarks[task][model_name].append((mean, config, name))

        for task, models in found_benchmarks.items():
            for name, all_configs in models.items():
                weight = self.weights[task] * (1.0/len(all_configs))
                for mean, config, benchmark in all_configs:
                    benchmark_score = weight * math.log(self.norm[benchmark] / mean)
                    score_db[config] += benchmark_score

        return score_db

    def compute_score(self, data):
        """
        This API calculates the total score for all the benchmarks
        that was run  by reading the data (.json) file.
        The weights are then calibrated to the target score.
        """
        score = 0.0
        score_db = self.get_score_per_config(data)
        score = sum(score_db.values())
        score = self.target * math.exp(score)
        return score

TORCHBENCH_V0_SCORE = TorchBenchScore(TORCHBENCH_V0_REF_DATA)
# TODO: Enable TORCHBENCH_V1_SCORE once TORCHBENCH_V1_REF_DATA is available
# TORCHBENCH_V1_SCORE = TorchBenchScore(ref_data=TORCHBENCH_V1_REF_DATA)
