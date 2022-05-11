"""
Compute TorchBench Score V2.
"""
import re
import math
import yaml
import importlib
import itertools
from pathlib import Path
from typing import List, Optional

TORCHBENCH_V2_REF_DATA = Path(__file__).parent.joinpath("configs/v2/config-v2.yaml")
TORCHBENCH_V2_DEFAULT_THRESHOLD = 0.07
TORCHBENCH_V2_DEFAULT_TARGET = 1000.0

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
    test, model_name, device, mode = re.match(r"test_(.*)\[(.*)\-(.*)\-(.*)\]", name).groups()
    return (test, model_name, device, mode)

class TorchBenchV2Test:
    def __init__(self, test_name, test_item):
        self._name = test_name
        self._test_type, self._model, self._device, self._mode = _parse_test_name(self._name)
        self._task = _get_model_task(self._model)
        self._stable = test_item["stable"]
        self._norm = test_item["norm"]
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
    def norm(self) -> float:
        return self._norm
    @property
    def stable(self) -> bool:
        return self._stable

class TorchBenchV2Suite:
    def __init__(self, norm):
        self._tests = []
        self._tests_dict = {}
        self._threshold = norm["stable_threshold"]
        self._target = norm["target"]
        for test in norm["tests"]:
            test_item = TorchBenchV2Test(test, norm["tests"][test])
            self._add_test(test_item)
    @property
    def target(self) -> float:
        return self._target
    @property
    def all_stable_tests(self) -> List[TorchBenchV2Test]:
        return list(filter(lambda x: x.stable, self._tests))
    @property
    def threshold(self) -> float:
        return self._threshold
    def _add_test(self, test: TorchBenchV2Test):
        self._tests.append(test)
        self._tests_dict[test.name] = test
    def get_test_by_name(self, name) -> TorchBenchV2Test:
        return self._tests_dict[name]

class TorchBenchScoreV2:
    # ref_data: the object read from reference YAML file or benchmark json file
    def __init__(self, ref_data, _spec_file, _target):
        if not ref_data:
            with open(TORCHBENCH_V2_REF_DATA) as ref_file:
                ref_data = yaml.full_load(ref_file)
        # Build the suite
        self.suite = TorchBenchV2Suite(self._setup_benchmark_norms(ref_data))

    def _get_test_delta_weight(self, ref_norm, data_norm):
        delta = (ref_norm - data_norm) / ref_norm
        # Not a valid signal because it is below threshold
        if abs(delta) <= self.suite.threshold:
            return 0.0
        return delta * 100

    def _get_delta_score(self, data_norm):
        "Compute V2 delta score"
        delta = 0.0
        for test in self.suite.all_stable_tests:
            ref_norm = test.norm
            data_test_norm = data_norm["tests"][test.name]["norm"]
            delta_weight = self._get_test_delta_weight(ref_norm, data_test_norm)
            delta += delta_weight
        return delta

    def _get_domain_score(self, data_norm, condition=None) -> Optional[float]:
        "Compute V2 domain subscore or total score"
        def _test_filter(test, condition) -> bool:
            # Total score, condition is None
            if not condition:
                return True
            device, test_type, domain = condition
            in_device = device in test.name
            in_type = test_type in test.name
            in_domain = test.domain in domain or test.category in domain
            return in_device and in_type and in_domain
        score = 0.0
        tests = self.suite.all_stable_tests
        filtered_tests = list(filter(lambda x: _test_filter(x, condition), tests))
        # Don't have any test in this category
        if not len(filtered_tests):
            return None
        # Each test has equal weight
        weight = 1.0 / len(filtered_tests)
        for test in filtered_tests:
            norm = data_norm["tests"][test.name]["norm"]
            delta = (norm - test.norm) / test.norm
            if abs(delta) <= self.suite.threshold:
                norm = test.norm
            score += weight * math.log(test.norm / norm)
        return math.exp(score) * self.suite.target

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
            ref_data = self._get_norm_from_ref_json_obj(ref_data)
        return ref_data

    def _get_norm_from_ref_json_obj(self, ref_json_obj):
        """
        This function iterates over the reference benchmark json output
        and calculates the normalization values based on the reference data.
        """
        norm = dict()
        norm["stable_threshold"] = TORCHBENCH_V2_DEFAULT_THRESHOLD
        norm["target"] = TORCHBENCH_V2_DEFAULT_TARGET
        norm["tests"] = dict()
        for b in ref_json_obj['benchmarks']:
            name = b['name']
            norm['tests'].setdefault(name, dict())
            norm['tests'][name]['norm'] = b['stats']['median']
            norm['tests'][name]['stable'] = True
        return norm

    def get_norm(self, data):
        return self._get_norm_from_ref_json_obj(data)

    def compute_score(self, data):
        """
        This API calculates the total V2 score for all the benchmark tests in the set.
        """
        def domain_to_condition(all_domains, domain):
            if domain == "OVERALL":
                return all_domains[1:]
            else:
                return [domain]
        # Check the input test set is the superset of the ref
        data_norm = self._get_norm_from_ref_json_obj(data)
        stable_tests = map(lambda x: x.name, self.suite.all_stable_tests)
        diff_set = set(stable_tests) - set(data_norm["tests"].keys())
        if diff_set:
            raise ValueError(f"The request benchmark json doesn't include the V2 test: {diff_set}")
        summary = {}
        # overall score
        summary["total"] = self._get_domain_score(data_norm)
        # delta score
        summary["delta"] = self._get_delta_score(data_norm)
        # domain scores
        summary["domain"] = {}
        axis_device = ["cuda", "cpu"]
        axis_test = ["train", "eval"]
        axis_domain = ["OVERALL", "NLP", "CLASSIFICATION", "SEGMENTATION", "SPEECH", "RECOMMENDATION"]
        for element in itertools.product(*[axis_device, axis_test, axis_domain]):
            dev, tp, domain = element
            cond = (dev, tp, domain_to_condition(axis_domain, domain))
            summary["domain"][f"{dev}-{tp}-{domain.lower()}"] = self._get_domain_score(data_norm, cond)
        return summary
