#!/bin/bash

set -eux

if [[ "$TEST_CONFIG" == "cpu" ]]; then
  python3 -m torchbenchmark._components.test.test_subprocess
  python3 -m torchbenchmark._components.test.test_worker
fi

# Test models
python3 test.py -v -k "$TEST_CONFIG"
