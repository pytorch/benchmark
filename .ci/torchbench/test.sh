#!/bin/bash

set -eux

pip install expecttest
# TO BE REMOVED
apt-get install -y libgl1-mesa-dev

if [[ "$TEST_CONFIG" == "cpu" ]]; then
  python3 -m torchbenchmark._components.test.test_subprocess
  python3 -m torchbenchmark._components.test.test_worker
fi

# Test models
python3 test.py -v -k "$TEST_CONFIG"
