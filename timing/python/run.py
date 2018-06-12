from __future__ import print_function

import os
import sys

sys.path.append(os.path.abspath("framework"))

from glob import glob
from benchmarks import NumpyComparison
from benchmarks import Convnets, LSTMBench
import framework

if __name__ == "__main__":
    framework.main(sys.argv, [NumpyComparison, Convnets, LSTMBench])
