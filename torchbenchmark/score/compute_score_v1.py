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
from typing import List

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

def _parse_test_name(name):
    """
    Helper function which extracts test type (eval or train), model, 
    device, and mode from the test full name.
    """
    if "-freeze" in name:
        name.replace("-freeze", "", 1)
    test, model_name, device, mode = re.match(r"test_(.*)\[(.*)\-(.*)\-(.*)\]", name).groups()
    return (test, model_name, device, mode)

class TorchBenchV1Test:
    def __init__(self, test_name):
        self._name = test_name
        self._test_type, self._model, self._device, self._mode = _parse_test_name(test_name)
        self._task = _get_model_task(self._model)
    @property
    def name(self) -> str:
        return self._name
    @property
    def test_type(self) -> str:
        return self._test_type
    @property
    def model(self) -> str:
        return self._model
    @property
    def device(self) -> str:
        return self._device
    @property
    def mode(self) -> str:
        return self._mode
    @property
    def category(self) -> str:
        return type(self._task).__name__
    @property
    def domain(self) -> str:
        return self._task.name
    @property
    def weight(self) -> float:
        # config weight rule in V1: 1x CPU Training, 2x GPU Training, 2x CPU Inference, 2x GPU Inference
        if self.test_type == "train" and self.device == "cpu":
            return 1.0
        return 2.0

class TorchBenchV1Suite:
    def __init__(self):
        self._suite_spec = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self._tests = []
    @property
    def all_tests(self):
        return self._tests
    def add_test(self, test: TorchBenchV1Test):
        self._suite_spec[test.category][test.domain][test.model].append(test)
        self._tests.append(test)
    def categories(self) -> List[str]:
        return self._suite_spec.keys()
    def domains(self, category: str) -> List[str]:
        return self._suite_spec[category].keys()
    def models(self, category: str, domain: str) -> List[str]:
        return self._suite_spec[category][domain].keys()
    def tests(self, category:str, domain: str, model: str) -> List[TorchBenchV1Test]:
        return self._suite_spec[category][domain][model]

class TorchBenchScoreV1:
    # ref_data: the YAML file or json file
    def __init__(self, ref_data, spec_file, target):
        if not ref_data:
            with open(TORCHBENCH_V1_REF_DATA) as ref_file:
                ref_data = yaml.full_load(ref_file)
        self.norm = self._setup_benchmark_norms(ref_data)
        self.norm_weights = self._setup_weights(self.norm)
        # spec_file is not used in V1, this is just a placeholder
        self.spec_file = spec_file
        self.target = target
        
    def _filter_jit_tests(self, norm):
        result_ref = dict()
        for jit_name in filter(lambda x: '-jit' in x, norm.keys()):
            left, sep, right = jit_name.rpartition('-jit')
            eager_name = left + "-eager" + right
            # We assume if a jit test exists, there must be an eager test
            assert eager_name in norm, f"Can't find eager test name {eager_test_name}"
            result_ref[jit_name] = dict()
            result_ref[jit_name]['jit_norm'] = norm[jit_name]['norm']
            result_ref[jit_name]['eager_norm'] = norm[eager_name]['norm']
        return result_ref

    # Generate the domain weights from the ref object
    def _setup_weights(self, ref):
        domain_weights = defaultdict(float)
        config_weights = defaultdict(float)
        # Build the test suite
        suite = TorchBenchV1Suite()
        for name in ref:
            test = TorchBenchV1Test(name)
            suite.add_test(test)
        # Setup domain weights
        for test in suite.all_tests:
            category_cnt = len(suite.categories())
            domain_cnt = len(suite.domains(test.category))
            model_cnt = len(suite.models(test.category, test.domain))
            domain_weights[test.name] = (1.0 / category_cnt) * (1.0 / domain_cnt) * (1.0 / model_cnt)
        # Setup config weights
        for test in suite.all_tests:
            category = test.category
            domain = test.domain
            model = test.model
            model_tests = suite.tests(test.category, test.domain, test.model)
            config_weights[test.name] = test.weight / sum(map(lambda x: x.weight, model_tests))
        # Runtime check the weights constraint
        sum_weight = 0.0
        for test in suite.all_tests:
            sum_weight += config_weights[test.name] * domain_weights[test.name]
        assert(abs(sum_weight - 1.0) < 1e-6), f"The total weights sum ({sum_weight}) is not 1.0, please submit a bug report."
        return (domain_weights, config_weights)

    def _setup_benchmark_norms(self, ref_data):
        """
        Helper function which gets the normalization values per benchmark
        by going through the reference data file.
        If ref_data is a benchmark json object, construct the YAML norm file from it.
        Otherwise, use it as-is.
        """
        assert isinstance(ref_data, dict), "The type of ref_data must be a dict object."
        # If the data contains machine_info key, it must be a benchmark json object
        if "benchmarks" in ref_data and "machine_info" in ref_data:
            ref = self._get_norm_from_ref_json_obj(ref_data)
        return ref

    def _get_norm_from_ref_json_obj(self, ref_json_obj):
        """
        This function iterates over the reference benchmark json output
        and calculates the normalization values based on the reference data.
        It also sets up the domain weights of the score.
        """
        norm = dict()
        for b in ref_json_obj['benchmarks']:
            norm.setdefault(b['name'], dict())
            norm[b['name']].setdefault('norm', dict())
            norm[b['name']]['norm'] = b['stats']['mean']
        return norm

    def _get_score(self, data, ref, ref_weights):
        score = 0.0
        (domain_weights, config_weights) = ref_weights
        for name in data:
            norm = data[name]['norm']
            benchmark_score = domain_weights[name] * config_weights[name] * math.log(norm / ref[name]['norm'])
            score += benchmark_score
        return math.exp(score)

    def data_in_list(self, n, l):
        for e in l:
            if e not in n:
                return False
        return True

    def _get_subscore(self, data, ref_norm, ref_weights, filters):
        error_msg = "We only accept one of the following four subscores: [cpu, train], [cpu, eval], [cuda, train], [cuda, infer]."
        assert len(filters) == 2, error_msg
        assert "cpu" in filters or "cuda" in filters, error_msg
        assert "train" in filters or "eval" in filters, error_msg
        score = 0.0
        (domain_weights, _) = ref_weights
        for name in filter(lambda x: self.data_in_list(x, filters), data):
            norm = data[name]['norm']
            benchmark_score = domain_weights[name] * math.log(norm / ref_norm[name]['norm'])
            score += benchmark_score
        return math.exp(score)
    
    def compute_jit_speedup_score(self, data):
        """
        This API calculates the V1 JIT speedup score for all 
        the benchmarks that enable JIT compilation.
        The data argument is the json data object from the benchmark.
        The JIT speedup score is the geometric mean of all JIT benchmarks speedup
        comparing to corresponding non-JIT benchmarks. Its computation does not require reference data.
        """
        score = 0.0
        norm = self._setup_benchmark_norms(data)
        norm_jit = self._filter_jit_tests(norm)
        (domain_weights, config_weights) = self._setup_weights(norm_jit)
        for name in norm_jit:
            eager_norm = norm_jit[name]['eager_norm']
            jit_norm = norm_jit[name]['jit_norm']
            jit_speedup_score = domain_weights[name] * config_weights[name] * math.log(eager_norm / jit_norm)
            score += jit_speedup_score
        return math.exp(score)

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
        data_norm = self._get_norm_from_ref_json_obj(data)
        for f in filters:
            key = f"subscore[{f[0]}-{f[1]}]"
            summary[key] = self._get_subscore(data_norm, self.norm, self.norm_weights, f) * self.target
        summary["total"] = self._get_score(data_norm, self.norm, self.norm_weights) * self.target
        return summary

    def get_norm(self, data):
        return self._get_norm_from_ref_json_obj(data)
