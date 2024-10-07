#!/bin/bash

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

# Test Tritonbench operators
# TODO: test every operator, fwd+bwd
python run_benchmark.py triton --op addmm --fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op flash_attention --fwd --num-inputs 1 --test-only

