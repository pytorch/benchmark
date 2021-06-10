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
    # ref_data: the YAML file or json file
    def __init__(self, ref_data, spec_file, target):
        self.norm = self._setup_benchmark_norms(ref_data)
        self.norm_weights = self._setup_weights(self.norm)
        self.norm_jit = _filter_jit_tests(self.norm)
        self.norm_jit_weights = self._setup_weights(self.norm_jit)
        # spec_file is not used in V1, this is just a placeholder
        self.spec_file = spec_file
        self.target = target
        
    def _filter_jit_tests(self, ref):
        result_ref = dict()
        result_ref['benchmarks'] = dict()
        for jit_name in ref['benchmarks'].keys().filter(lambda x: '-jit' in x):
            left, sep, right = jit_name.rpartition('-jit')
            eager_name = left + "-eager" + right
            # We assume if a jit test exists, there must be an eager test
            assert eager_name in ref['benchmarks'], f"Can't find eager test name {eager_test_name}"
            result_ref['benchmarks'][jit_name]['jit_norm'] = ref['benchmarks'][jit_name].copy()
            result_ref['benchmarks'][jit_name]['eager_norm'] = ref['benchmarks'][eager_name].copy()
            result_ref['benchmarks'][jit_name]['norm'] = ref['benchmarks'][jit_name]['norm'] / ref['benchmarks'][eager_name]['norm'] 
        return result_ref

    def _test_to_config(self, test):
        if "-freeze" in test:
            test.replace("-freeze", "", 1)
        test, model_name, device, mode = re.match(r"test_(.*)\[(.*)\-(.*)\-(.*)\]", name).groups()
        return (test, model_name, device, mode)

    def _config_to_weight(self, config):
        test, model_name, device, mode = config
        if test == "train" and device == "cpu":
            return 1.0
        else:
            return 2.0
        
    # Generate the domain weights from the ref object
    def _setup_weights(self, ref):
        # Setup domain_weights
        domain_weights = {}
        config_weights = {}
        config_dict = defaultdict()
        task_dict = defaultdict()
        name_dict = defaultdict()
        for b in ref_data['benchmarks']:
            name = b['name']
            task = _get_model_task(name)
            config = self._test_to_config(name)
            task_dict[type(task)][task] += 1
            config_dict[config.2][name] = self._config_to_weight(config)
            name_dict[name] = task
        category_cnt = len(task_dict)
        for name in name_dict:
            task = name_dict[name]
            domain_cnt = len(task_dict[type(task)])
            task_cnt = domain_dict[type(task)][task]
            self.domain_weights[name] = (1.0 / category_cnt) * (1.0 / domain_cnt) * (1.0 / task_cnt)
        # Setup config_weights
        for name in name_dict:
            (test, model_name, device, mode) = self._test_to_config(name)
            config_weights[name] = config_dict[model_name][name] / sum(config_dict[model_name])
        # config weight rule in V1: 1x CPU Training, 2x GPU Training, 2x CPU Inference, 2x GPU Inference
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
            ref = self._get_ref_from_ref_data(self.ref_data)
        return ref

    def _get_ref_from_ref_data(self, ref_data):
        """
        This function iterates over the reference data (json object)
        and calculates the normalization values based on the reference data.
        It also sets up the domain weights of the score.
        """
        ref = {}
        ref['benchmarks'] = {}
        for b in ref_data['benchmarks']:
            d = {}
            d['norm'] = b['stats']['mean']
            ref['benchmarks'][b['name']] = d
        return ref

    def _get_score(self, data, ref, ref_weights):
        score = 0.0
        (domain_weights, config_weights) = ref_weights
        for name in data['benchmarks']:
            norm = data['benchmarks'][name]['norm']
            benchmark_score = domain_weights[name] * config_weights[name] * math.log(norm / ref['benchmarks'][name]['norm'])
            score += benchmark_score
        return math.exp(score)
        
    def _get_subscore(self, data, ref_data, ref_weights, filter_lambda):
        score = 0.0
        (domain_weights, _) = ref_weights
        for name in data['benchmarks'].filter(filter_lambda):
            norm = data['benchmarks']['norm']
            benchmark_score = domain_weights[name] * math.log(norm / ref['benchmarks'][name]['norm'])
            score += benchmark_score
        return math.exp(score)
    
    def compute_jit_speedup_score(self, data):
        # Assert the jitonly_data has the same set of tests as self.ref_jit_data
        score = self._get_score(data, self.ref_jit_data, self.ref_jit_weights)
        return score
    
    def compute_score(self, data, filter_str = ""):
        """
        This API calculates the total V0 score for all the
        benchmarks that was run by reading the data (.json) file.
        The weights are then calibrated to the target score.
        """
        allowed_filter = ["", "cpu-eval", "cpu-train", "cuda-eval", "cuda-train"]
        data_ref = self._get_ref_from_ref_data(data)
        assert filter_str in allowed_filter, f"We don't allow subscore filter {filter_str}."
        if filter_str:
            filter_lambda = 
            score = self._get_subscore(data, self.ref_data, self.ref_weights, filter_lambda)
        else:
            score = self._get_score(data_ref, self.ref_data, self.ref_weights)
        return score

    def get_norm(self, jit=False):
        if jit:
            return self.norm_jit_data
        return self.norm
