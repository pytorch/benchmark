#!/bin/bash

PLATFORM="${PLATFORM:-}"
COMMAND_ARGS="${COMMAND_ARGS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
if [[ $COMMAND_ARGS == *"collect-only"* ]]
then
    pytest ./test_bench.py $COMMAND_ARGS 2>&1 |& tee $OUTPUT_DIR/list_test_bench.txt
else
    mkdir -p $OUTPUT_DIR/$PLATFORM
    file_name=test_bench_"$MODE"_"$(date +%Y%m%d_%H%M%S)".txt
    echo "* Running: pytest ./test_bench.py" $COMMAND_ARGS 2>&1 |& tee -a $OUTPUT_DIR/$PLATFORM/$file_name
    pytest ./test_bench.py $COMMAND_ARGS 2>&1 |& tee -a $OUTPUT_DIR/$PLATFORM/$file_name
    cp -Rf ./.benchmarks/Linux-CPython-3.7-64bit/* $OUTPUT_DIR/$PLATFORM
fi
