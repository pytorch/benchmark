#!/bin/bash

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

# Test Tritonbench operators
# TODO: test every operator, fwd+bwd
python run_benchmark.py triton --op launch_latency --fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op addmm --fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op gemm --fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op sum --fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op softmax --fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op layer_norm --fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op flash_attention --fwd --num-inputs 1 --test-only

python run_benchmark.py triton --op jagged_layer_norm --fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op jagged_mean --fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op jagged_softmax --fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op jagged_sum --fwd --num-inputs 1 --test-only
