from __future__ import print_function

import os
import sys

sys.path.append(os.path.abspath("framework"))

from benchmarks import CPUConvnets
from benchmarks import CPULSTMBench
from benchmarks import CPUNNBench
from benchmarks import CUDALSTMBench
from benchmarks import NumpyUnaryComparison
from benchmarks import CPUUnaryBench
from benchmarks import NumpyReduceComparison

import framework

if __name__ == "__main__":
    framework.main(
        sys.argv,
        [
            CPUConvnets,
            CPULSTMBench,
            CPUNNBench,
            CPUUnaryBench,
            CUDALSTMBench,
            NumpyReduceComparison,
            NumpyUnaryComparison,
        ],
    )
