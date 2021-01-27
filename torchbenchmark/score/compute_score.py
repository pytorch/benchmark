
"""
Compute the benchmark score given a frozen score configuration and current benchmark data.
"""
import argparse
import json
import math
import sys
import os
import yaml

from tabulate import tabulate

SPEC_FILE_DEFAULT = "score.yml"
TARGET_SCORE_DEFAULT = 1000

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

