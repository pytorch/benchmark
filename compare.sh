#!/bin/bash
rm -f old.json te.json
pytest test_bench.py -k cuda-jit --fuser old --benchmark-json old.json
pytest test_bench.py -k cuda-jit --fuser te --benchmark-json te.json
python compare.py old.json te.json
