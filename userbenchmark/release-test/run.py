import os
import subprocess

from typing import List

BM_NAME = "release-test"
EXAMPLE_URL = "https://github.com/pytorch/examples.git"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(args: List[str]):
    subprocess.check_call(["bash", f"{CURRENT_DIR}/run_release_test.sh"])
