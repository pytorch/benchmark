#!/bin/bash

file_name=test_bench_all_local_"$(date +%Y%m%d_%H%M%S)".txt
echo "* Running: pytest ./test_bench.py --ignore_machine_config --benchmark-autosave --cpu_only" 2>&1 |& tee -a ./cpu_output/$file_name
pytest ./test_bench.py --ignore_machine_config --benchmark-autosave --cpu_only 2>&1 |& tee -a ./cpu_output/$file_name
cp -Rf ./.benchmarks/* ./cpu_output/cpu/
