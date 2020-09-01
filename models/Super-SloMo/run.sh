#!/bin/bash
python train.py --dataset_root dataset --checkpoint_dir checkpoints --epochs 1 "$@"
