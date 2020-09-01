#! /bin/bash
set -e

time bash run.sh --debug reference_0.out
time bash run.sh --debug reference_1.out
python check.py reference_0.out reference_1.out

time bash run.sh --script --debug jit.out
python check.py reference_0.out jit.out
