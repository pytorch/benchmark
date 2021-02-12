
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

SPEC_FILE_DEFAULT = "torchbenchmark/score/score.yml"
TARGET_SCORE_DEFAULT = 1000

class TorchBenchScore:
    def __init__(self, spec=SPEC_FILE_DEFAULT, target=TARGET_SCORE_DEFAULT):
        self.spec = spec
        self.target = target
        self.weights = None
        self.norm = None

    def setup_weights(self):
        """
        Calculates the static benchmark weights by iterating the spec
        file and constructs a dictionary with (key, value) pair
        is (task, weight_for_benchmark_per_task)

        The spec here can be user defined .yaml file or if no spec is defined,
        the default score.yaml is used.
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

    def setup_benchmark_norms(self, data):
        """
        Helper function which gets the normalization values per benchmark
        by going through the reference data file.

        This reference data can be user provided or by default the first file
        in the directory of data files is considered as the reference data file
        """
        self.norm = {b['name']: b['stats']['mean'] for b in data['benchmarks']}

    def _get_model_task(self, model_name):
        """
        Helper function which extracts the task the model belongs to
        by iterating over the Model attributes.
        """
        MODEL_PATH="../../models"
        p = Path(__file__).parent.joinpath(MODEL_PATH + model_name)
        if(p):
            module = importlib.import_module(f'torchbenchmark.models.{model_name}', package=__name__)
            Model = getattr(module, 'Model')
        return Model.task.value

    def _get_benchmark_configuration(self, b):
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
        if "eval" in b: test = "eval"
        if "cpu" in b: device = "cpu"
        if "jit" in b: mode = "jit"

        # Extract the model name from the benchmark b
        model_name = (re.findall(r'\[(.*)\-'+device, b))[0]

        # Get the Model task value by reading the Model attributes.
        task = self._get_model_task(model_name)
        return (test, device, mode), task, model_name

    def _construct_benchmark_run_db(self, data):
        """
        Construct a benchmark database by going over through the data file
        for the run and update the dictionary by task and model_name

        For eg., the (key, value) for this dictionary is of the form
        [generation][pytorch_stargan] = [(1.2, (eval, cuda, jit),
                                        test_eval[pytorch_stargan-cuda-jit])]
        """
        found_benchmarks = defaultdict(lambda: defaultdict(list))

        for b in data['benchmarks']:
            name, mean = b['name'], b['stats']['mean']
            config, task, model_name = self._get_benchmark_configuration(name)
            # Append the tuple(mean, config, model_name) for all the configs the
            # benchmark was run with.
            found_benchmarks[task][model_name].append((mean, config, name))
        return found_benchmarks

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
        found_benchmarks = self._construct_benchmark_run_db(data)
        score_db = defaultdict(float)

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
        if self.norm is None:
            self.setup_benchmark_norms(data)
        if self.weights is None:
            self.setup_weights()

        score = 0.0
        score_db = self.get_score_per_config(data)
        score = sum(score_db.values())
        score = self.target * math.exp(score)
        return score
