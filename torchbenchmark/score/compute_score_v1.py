
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
TORCHBENCH_V1_REF_DATA = Path(__file__).parent.joinpath("configs/v1/config-v1.yaml")

def _get_model_task(model_name):
    """
    Helper function which extracts the task the model belongs to
    by iterating over the Model attributes.
    """
    try:
        module = importlib.import_module(f'torchbenchmark.models.{model_name}', package=__name__)
    except:
        raise ValueError(f"Unable to get task for model: {model_name}")
    Model = getattr(module, 'Model')
    return Model.task.value

class TorchBenchScoreV1:
    def __init__(self, ref_data, target=TARGET_SCORE_DEFAULT):
        self.target = target
        self.ref_data = ref_data
        self.norm = None

        self._setup_benchmark_norms()

    # Generate reference data from json object
    def _generate_ref(self, ref_data):
        pass
        
    def _setup_benchmark_norms(self):
        """
        Helper function which gets the normalization values per benchmark
        by going through the reference data file.
        """
        if self.ref_data == TORCHBENCH_V1_REF_DATA:
            with open(self.ref_data) as ref_file:
                ref = yaml.full_load(ref_file)
        else:
            ref = self._generate_ref(self.ref_data)
        self.norm = ref['benchmarks']

    def get_score_per_config(self, data, weighted_score=False):
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

        # Get the weights per config and calibrate it to the
        # target score
        if weighted_score:
            for config, score in score_db.items():
                score_db[config] = score * 0.125
                score_db[config] = self.target * math.exp(score)

        return score_db

    def compute_score(self, data):
        """
        This API calculates the total V0 score for all the
        benchmarks that was run by reading the data (.json) file.
        The weights are then calibrated to the target score.
        """
        score = 0.0
        score_db = self.get_score_per_config(data)
        score = sum(score_db.values())
        score = self.target * math.exp(score)
        return score

    def compute_score_v1(self, data):
        """
        This API calculates the total V1 score for all the 
        benchmarks that was run by reading the data (.json) file.
        The weights are then calibrated to the target score.
        """
        score = 0.0
        pass

TORCHBENCH_V1_SCORE = TorchBenchScore(ref_data=TORCHBENCH_V1_REF_DATA)
