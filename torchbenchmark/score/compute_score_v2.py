"""
Compute TorchBench Score V2.
"""
import re
import yaml
import importlib
from pathlib import Path
from typing import List

TORCHBENCH_V2_REF_DATA = Path(__file__).parent.joinpath("configs/v2/config-v2.yaml")
TORCHBENCH_V2_DEFAULT_THRESHOLD = 0.07

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
    def __init__(self, test_item):
        self._name = test_item.keys()[0]
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
        for test in norm["tests"]:
            test_item = TorchBenchV2Test(test)
            self._add_test(test_item)
    @property
    def all_stable_tests(self) -> List[TorchBenchV2Test]:
        return list(filter(lambda x: x.stable, self._tests))
    @property
    def threshold(self) -> float:
        return self.threshold
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
        delta = (ref_norm - data_norm) / ref_norm * 100.0
        # Not a valid signal because it is below threshold
        if abs(delta) <= self.suite.threshold:
            return 0.0
        return delta / 100.0

    # compute the V2 total score
    def _get_delta_score(self, data_norm):
        delta = 0.0
        for test in self.suite.all_stable_tests:
            ref_norm = test.norm
            data_test_norm = data_norm["tests"][test.name]["norm"]
            delta_weight = self._get_test_delta_weight(ref_norm, data_test_norm)
            delta += delta_weight
        return delta

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
        # Check the input test set is the superset of the ref
        data_norm = self._get_norm_from_ref_json_obj(data)
        diff_set = set(self.norm.keys()) - set(data_norm.keys())
        assert not diff_set, f"The request benchmark json doesn't include the V2 test: {diff_set}"
        summary = {}
        # overall score
        # delta score
        summary["delta"] = self._get_delta_score(data_norm)
        # domain score
        return summary
