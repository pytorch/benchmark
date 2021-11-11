#!/bin/bash

MODE="${MODE:-}"
COMMAND_ARGS="${COMMAND_ARGS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
echo input:
echo $COMMAND_ARGS
if [[ $COMMAND_ARGS == *"collect-only"* ]]
then
    pytest test.py $COMMAND_ARGS 2>&1 | tee $OUTPUT_DIR/list_test.txt
else
    file_name=test_"$MODE"_"$(date +%Y%m%d_%H%M%S)".txt
    echo running python test.py $COMMAND_ARGS > $OUTPUT_DIR/$file_name
    python test.py $COMMAND_ARGS 2>&1 |& tee -a $OUTPUT_DIR/$file_name
    echo DONE
fi