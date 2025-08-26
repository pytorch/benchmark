#!/bin/bash

set -eux

pip install expecttest

# TODO (huydhn): Not sure why the worker swallows up SIGSEGV when running
# on CPU
if [[ "$TEST_CONFIG" == "cuda" ]]; then
  python3 -m torchbenchmark._components.test.test_subprocess
  python3 -m torchbenchmark._components.test.test_worker
fi

# Test models
python3 test.py -v -k "$TEST_CONFIG"
