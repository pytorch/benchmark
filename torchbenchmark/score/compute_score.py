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

from .compute_score_v0 import TorchBenchScoreV0, TORCHBENCH_V0_REF_DATA
from .compute_score_v1 import TorchBenchScoreV1, TORCHBENCH_V1_REF_DATA

TARGET_SCORE_DEFAULT = 1000
SPEC_FILE_DEFAULT = Path(__file__).parent.joinpath("score.yml")

class TorchBenchScore:
    def __init__(self, ref_data=TORCHBENCH_V1_REF_DATA, spec=SPEC_FILE_DEFAULT,
                 target=TARGET_SCORE_DEFAULT, version="v1"):
        active_versions = {"v0": TorchBenchScoreV0, "v1": TorchBenchScoreV1 }
        if version not in active_versions:
            print(f"We only support TorchBench score versions: {active_versions.keys()}")
        self.score = active_versions[version](ref_data, spec, target)
    
    def compute_score(self, data):
        return self.score.compute_score(data)
