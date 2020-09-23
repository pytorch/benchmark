#!/bin/bash

echo "Model Size Check"
du -h torchbenchmark/models --max-depth=1 | sort -h