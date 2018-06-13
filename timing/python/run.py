from __future__ import print_function

import os
import sys

sys.path.append(os.path.abspath("framework"))

from benchmarks import NumpyComparison
from benchmarks import CPUConvnets
from benchmarks import CUDALSTMBench
from benchmarks import CPULSTMBench
import framework

if __name__ == "__main__":
    framework.main(
        sys.argv, [NumpyComparison, CPUConvnets, CUDALSTMBench, CPULSTMBench]
    )
