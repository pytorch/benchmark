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

from enum import Enum
from tabulate import tabulate
from pathlib import Path
from collections import defaultdict

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
    return Model.task

class TorchBenchScoreV1:
    # ref_data: the YAML file or json file
    def __init__(self, ref_data, spec_file, target):
        if not ref_data:
            ref_data = TORCHBENCH_V1_REF_DATA
        self.norm = self._setup_benchmark_norms(ref_data)
        self.norm_weights = self._setup_weights(self.norm)
        # spec_file is not used in V1, this is just a placeholder
        self.spec_file = spec_file
        self.target = target
        
    def _filter_jit_tests(self, norm):
        result_ref = dict()
        result_ref['benchmarks'] = dict()
        for jit_name in filter(lambda x: '-jit' in x, norm['benchmarks'].keys()):
            left, sep, right = jit_name.rpartition('-jit')
            eager_name = left + "-eager" + right
            # We assume if a jit test exists, there must be an eager test
            assert eager_name in norm['benchmarks'], f"Can't find eager test name {eager_test_name}"
            result_ref['benchmarks'][jit_name] = dict()
            result_ref['benchmarks'][jit_name]['jit_norm'] = norm['benchmarks'][jit_name]['norm']
            result_ref['benchmarks'][jit_name]['eager_norm'] = norm['benchmarks'][eager_name]['norm']
        return result_ref

    def _test_to_config(self, name):
        if "-freeze" in name:
            name.replace("-freeze", "", 1)
        test, model_name, device, mode = re.match(r"test_(.*)\[(.*)\-(.*)\-(.*)\]", name).groups()
        return (test, model_name, device, mode)

    def _config_to_weight(self, config):
        # config weight rule in V1: 1x CPU Training, 2x GPU Training, 2x CPU Inference, 2x GPU Inference
        test, model_name, device, mode = config
        if test == "train" and device == "cpu":
            return 1.0
        else:
            return 2.0
        
    # Generate the domain weights from the ref object
    def _setup_weights(self, ref):
        # Setup domain_weights
        domain_weights = defaultdict(float)
        config_weights = defaultdict(float)
        config_dict = defaultdict(lambda: defaultdict(int))
        task_dict = defaultdict(lambda: defaultdict(set))
        name_dict = defaultdict(lambda: defaultdict(Enum))
        for name in ref['benchmarks']:
            config = self._test_to_config(name)
            task = _get_model_task(config[1])
            # print(f"Found domain {type(task).__name__}, task: {task.name} in model {name}")
            task_dict[type(task).__name__][task.name].add(config[1])
            config_dict[config[1]][name] = self._config_to_weight(config)
            name_dict[name] = task
        category_cnt = len(task_dict)
        sum_domain = 0.0
        for name in name_dict:
            task = name_dict[name]
            domain_cnt = len(task_dict[type(task).__name__])
            task_cnt = len(task_dict[type(task).__name__][task.name])
            domain_weights[name] = (1.0 / category_cnt) * (1.0 / domain_cnt) * (1.0 / task_cnt)
        # Setup config_weights
        for name in name_dict:
            (test, model_name, device, mode) = self._test_to_config(name)
            config_weights[name] = config_dict[model_name][name] / sum(config_dict[model_name].values())
        for name in name_dict:
            total_weight = config_weights[name] * domain_weights[name]
            sum_domain += total_weight
        assert(abs(sum_domain - 1.0) < 1e-6), "The total task weights sum is not 1.0, please submit a bug report."
        return (domain_weights, config_weights)
 
    def _setup_benchmark_norms(self, ref_data):
        """
        Helper function which gets the normalization values per benchmark
        by going through the reference data file.
        """
        if ref_data == TORCHBENCH_V1_REF_DATA:
            with open(ref_data) as ref_file:
                ref = yaml.full_load(ref_file)
        else:
            ref = self._get_norm_from_ref_data(ref_data)
        return ref

    def _get_norm_from_ref_data(self, ref_data):
        """
        This function iterates over the reference data (json object)
        and calculates the normalization values based on the reference data.
        It also sets up the domain weights of the score.
        """
        norm = {}
        norm['benchmarks'] = {}
        for b in ref_data['benchmarks']:
            d = {}
            d['norm'] = b['stats']['mean']
            norm['benchmarks'][b['name']] = d
        return norm

    def _get_score(self, data, ref, ref_weights):
        score = 0.0
        (domain_weights, config_weights) = ref_weights
        for name in data['benchmarks']:
            norm = data['benchmarks'][name]['norm']
            benchmark_score = domain_weights[name] * config_weights[name] * math.log(norm / ref['benchmarks'][name]['norm'])
            score += benchmark_score
        return math.exp(score)

    def data_in_list(self, n, l):
        for e in l:
            if e not in n:
                return False
        return True

    def _get_subscore(self, data, ref_norm, ref_weights, filters):
        assert len(filters) == 2, error_msg
        assert "cpu" in filters or "cuda" in filters, error_msg
        assert "train" in filters or "eval" in filters, error_msg
        score = 0.0
        (domain_weights, _) = ref_weights
        for name in filter(lambda x: self.data_in_list(x, filters), data['benchmarks']):
            norm = data['benchmarks'][name]['norm']
            benchmark_score = domain_weights[name] * math.log(norm / ref_norm['benchmarks'][name]['norm'])
            score += benchmark_score
        return math.exp(score)
    
    def compute_jit_speedup_score(self, data):
        """
        This API calculates the V1 JIT speedup score for all 
        the benchmarks that enable JIT compilation.
        The data argument is the json data object from the benchmark.
        The JIT speedup score is the geometric mean of all JIT benchmarks speedup
        comparing to corresponding non-JIT benchmarks.
        The weights are then calibrated to the target score.
        """
        score = 0.0
        norm = self._setup_benchmark_norms(data)
        norm_jit = self._filter_jit_tests(norm)
        (domain_weights, config_weights) = self._setup_weights(norm_jit)
        for name in norm_jit['benchmarks']:
            eager_norm = norm_jit['benchmarks'][name]['eager_norm']
            jit_norm = norm_jit['benchmarks'][name]['jit_norm']
            jit_speedup_score = domain_weights[name] * config_weights[name] * math.log(eager_norm / jit_norm)
            score += jit_speedup_score
        return math.exp(score) * self.target

    def compute_score(self, data):
        """
        This API calculates the total V1 score for all the
        benchmarks that was run by reading the data (.json) file.
        """
        summary = {}
        summary["subscore[jit]"] = self.compute_jit_speedup_score(data)
        devices = ["cpu", "cuda"]
        tests = ["train", "eval"]
        filters = [(a, b) for a in devices for b in tests]
        data_norm = self._get_norm_from_ref_data(data)
        for f in filters:
            key = f"subscore[{f[0]}-{f[1]}]"
            summary[key] = self._get_subscore(data_norm, self.norm, self.norm_weights, f) * self.target
        summary["total"] = self._get_score(data_norm, self.norm, self.norm_weights) * self.target
        return summary

    def get_norm(self, data):
        return self._get_norm_from_ref_data(data)
