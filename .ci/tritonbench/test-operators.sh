#!/bin/bash
set -x

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

# Test Tritonbench operators
# TODO: test every operator, fwd+bwd
python run_benchmark.py triton --op launch_latency --mode fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op addmm --mode fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op gemm --mode fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op sum --mode fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op softmax --mode fwd --num-inputs 1 --test-only
python run_benchmark.py triton --op layer_norm --mode fwd --num-inputs 1 --test-only


# Segfault
# python run_benchmark.py triton --op flash_attention --mode fwd --num-inputs 1 --test-only

# CUDA OOM
# python run_benchmark.py triton --op jagged_layer_norm --mode fwd --num-inputs 1 --test-only
# python run_benchmark.py triton --op jagged_mean --mode fwd --num-inputs 1 --test-only
# python run_benchmark.py triton --op jagged_softmax --mode fwd --num-inputs 1 --test-only
# python run_benchmark.py triton --op jagged_sum --mode fwd --num-inputs 1 --test-only
