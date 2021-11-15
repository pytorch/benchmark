#!/bin/bash

PLATFORM="${PLATFORM:-}"
MODE="${MODE:-}"
COMMAND_ARGS="${COMMAND_ARGS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
echo $(pwd)
if [[ $COMMAND_ARGS == *"collect-only"* ]]
then
    pytest ./test.py $COMMAND_ARGS 2>&1 | tee $OUTPUT_DIR/list_test.txt
else
    mkdir -p $OUTPUT_DIR/$PLATFORM
    file_name=test_"$MODE"_"$(date +%Y%m%d_%H%M%S)".txt
    echo "* Running: python ./test.py" $COMMAND_ARGS 2>&1 |& tee -a $OUTPUT_DIR/$PLATFORM/$file_name
    python ./test.py $COMMAND_ARGS 2>&1 |& tee -a $OUTPUT_DIR/$PLATFORM/$file_name
fi
