#!/usr/bin/env python3

# TODO
# Image-net args
    # Default train / val folder
# extract accuracy


# export PYTORCH_EXAMPLES_HOME=/Users/krishnakalyan3/open-source/torch_perf/examples/
import os
from os.path import join
from datetime import datetime
from tqdm import tqdm
import itertools
import sys
import argparse
import torchvision.models as models
import gc
import logging

model_names = sorted(name for name in models.__dict__
                     if not (not (name.islower() and not name.startswith("__")) or not callable(models.__dict__[name])))

parser = argparse.ArgumentParser(description="PyTorch model accuracy benchmark.")
parser.add_argument('--repeat',   type=int, default=1,
                    help="Number of Runs")
parser.add_argument('--arch',     type=str, default='all', choices=model_names,
                    help='model architectures: ' + ' | '.join(model_names) + ' (default: all)')
parser.add_argument('--temp-dir', type=str, default='log',
                    help='the path on the file system to place the working temporary directory at')
parser.add_argument('--filename', type=str, default='perf_test',
                    help='name of the output file')


def get_env_pytorch_examples():
    pytorch_examples_home = os.environ.get('PYTORCH_EXAMPLES_HOME')
    if pytorch_examples_home is None:
        print('PYTORCH_EXAMPLES_HOME not found')
        sys.exit()

    return pytorch_examples_home


def init(args):
    folder_root = os.getcwd()
    temp_dir = join(folder_root, args.temp_dir)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # logging
    log_filename = args.filename + '.out'
    logging.basicConfig(filename=join(temp_dir, log_filename), level=20)


def execution(cmd, iteration, model):
    gc.collect()
    accuracy = 0.95
    #os.system(cmd)
    return (model, accuracy, iteration)


def main():
    args = parser.parse_args()
    init(args)

    time_now = str(datetime.now())
    logging.info('New performance test started at {}'.format(time_now))
    logging.info('model, accuracy, iteration')

    pytorch_examples_home = get_env_pytorch_examples()

    # Build configuration
    k =itertools.product(['alex_net','image_net'], 'm')
    print(list(k))


if __name__ == '__main__':
    main()
