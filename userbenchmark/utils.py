import os
import sys
from datetime import datetime
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent
USERBENCHMARK_OUTPUT_PREFIX = ".userbenchmark"

PLATFORMS = [
    "gcp_a100",
    "aws_t4_metal",
]


class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


@dataclass
class TorchBenchABTestMetric:
    control: float
    treatment: float
    delta: float


@dataclass
class TorchBenchABTestResult:
    control_env: Dict[str, str]
    treatment_env: Dict[str, str]
    bisection: Optional[str]
    details: Dict[str, TorchBenchABTestMetric]


def get_output_json(bm_name, metrics):
    import torch
    return {
        "name": bm_name,
        "environ": {"pytorch_git_version": torch.version.git_version},
        "metrics": metrics,
    }

def dump_output(bm_name, output):
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    target_dir = current_dir.parent.joinpath(USERBENCHMARK_OUTPUT_PREFIX, bm_name)
    target_dir.mkdir(exist_ok=True, parents=True)
    fname = "metrics-{}.json".format(datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S"))
    full_fname = os.path.join(target_dir, fname)
    with open(full_fname, 'w') as f:
        json.dump(output, f, indent=4)

def get_output_dir(bm_name):
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    target_dir = current_dir.parent.joinpath(".userbenchmark", bm_name)
    target_dir.mkdir(exist_ok=True, parents=True)
    return target_dir