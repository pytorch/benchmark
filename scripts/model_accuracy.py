#!/usr/bin/env python3

import os
import argparse
from os.path import join
from datetime import datetime
import logging
from tqdm import tqdm
from utils import get_env_pytorch_examples, config_runs, cmd_string, execution, model_names


parser = argparse.ArgumentParser(description="PyTorch model accuracy benchmark.")
parser.add_argument('--repeat',   type=int, default=1,
                    help="Number of Runs")
parser.add_argument('--arch',     type=str, default='all', choices=model_names,
                    help='model architectures: ' + ' | '.join(model_names) + ' (default: all)')
parser.add_argument('--temp-dir', type=str, default='log',
                    help='the path on the file system to place the working temporary directory at')
parser.add_argument('--filename', type=str, default='perf_test',
                    help='name of the output file')
parser.add_argument('--data', type=str, required=True,
                    help='path to dataset')


def log_init():
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # logging
    log_filename = args.filename + '.out'
    logging.basicConfig(filename=join(temp_dir, log_filename), level=20)


def main():
    global args, temp_dir
    args = parser.parse_args()

    # temp dir
    folder_root = os.getcwd()
    temp_dir = join(folder_root, args.temp_dir)

    log_init()

    time_now = str(datetime.now())
    logging.info('New performance test started at {}'.format(time_now))
    logging.info('model, accuracy, iteration')

    examples_home = get_env_pytorch_examples()
    imagenet = join(examples_home, 'imagenet', 'main.py')

    runs = config_runs(args.arch, args.repeat)

    for i in tqdm(range(len(runs))):
        model, current_iter = runs[i]

        # logging execution
        file_name = runs[i][0] + str(runs[i][1]) + '.txt'
        log_path = join(temp_dir, file_name)

        # execution
        cmd = cmd_string(imagenet, model, args.data)
        accuracy = execution(cmd, log_path)

        logging.info('{},{},{}'.format(model, accuracy, current_iter))

if __name__ == '__main__':
    main()
