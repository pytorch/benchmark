
"""
Compute the benchmark score given a frozen score configuration and current benchmark data.
"""
import argparse
import json
import math
import sys
import os
import yaml

from collections import defaultdict
from tabulate import tabulate
from torchbenchmark import list_models

<<<<<<< HEAD
=======
#SPEC_FILE_DEFAULT = "score.yml"

>>>>>>> fa9c3b67367e2cabd3f45ef767b1f10fa8169e93
def generate_spec():
    """
    Helper function which constructs the spec dictionary by iterating
    over the existing models. This API is used for generating the
    default spec hierarchy configuration.
    return type:
        Returns a dictionary of type collections.defaultdict. Defaultdict
        handles `None` or missing values and hence, no explicit `None` check
        is required.
    Arguments: `None`
<<<<<<< HEAD
    Note: Only those models available inside the required_models list are
          used to construct the default spec hierarchy.
=======
    Note: Only those models with `domain` and `task` class attributes are
          used to construct the spec hierarchy.
>>>>>>> fa9c3b67367e2cabd3f45ef767b1f10fa8169e93
    """
    spec = {'hierarchy':{'model':defaultdict(dict)}}
    # These are the models required to generate the default spec.
    # Please update this list if any new model needs to be a part of
<<<<<<< HEAD
    # the default spec configuration. Before you update, please consult
    # with someone in the benchmark team.
=======
    # the default spec configuration.
>>>>>>> fa9c3b67367e2cabd3f45ef767b1f10fa8169e93
    required_models = ['pytorch_mobilenet_v3', 'yolov3', 'attention_is_all_you_need_pytorch', \
                       'BERT_pytorch', 'fastNLP', 'dlrm', 'LearningToPaint', 'moco', 'demucs', \
                       'pytorch_struct']

    for model in list_models():
        if model.name in required_models and hasattr(model, 'domain'):
            if not spec['hierarchy']['model'][model.domain]:
                spec['hierarchy']['model'][model.domain] = defaultdict(dict)
            if not spec['hierarchy']['model'][model.domain][model.task]:
                spec['hierarchy']['model'][model.domain][model.task] = defaultdict(dict)
            spec['hierarchy']['model'][model.domain][model.task][model.name] = None

    assert len(spec) == 0, f"Spec is empty. Make sure to use models only in the required list."
    return spec

def compute_score(config, data, fake_data=None):
    target = config['target']
    score = 0.0
    weight_sum = 0.0
    for name in config['benchmarks']:
        cfg = config['benchmarks'][name]
        weight, norm = cfg['weight'], cfg['norm']
        weight_sum += weight
        measured_mean = [b['stats']['mean'] for b in data['benchmarks'] if b['name'] == name]
        assert len(measured_mean) == 1, f"Missing data for {name}, unable to compute score"
        measured_mean = measured_mean[0]
        if fake_data is not None and name in fake_data:
            # used for sanity checks on the sensitivity of the score metric
            measured_mean = fake_data[name]
        benchmark_score = weight * math.log(norm / measured_mean)
        # print(f"{name}: {benchmark_score}")
        score += benchmark_score

    score = target * math.exp(score)
    assert abs(weight_sum - 1.0) < 1e-6, f"Bad configuration, weights don't sum to 1, but {weight_sum}"
    return score
