#!/usr/bin/env python3

import os
import argparse
from os.path import join
from datetime import datetime
import logging
from tqdm import tqdm
import gc
import itertools
import sys
import shlex
import subprocess
import torchvision.models as models


parser = argparse.ArgumentParser(description="PyTorch model accuracy benchmark.")
parser.add_argument('--repeat', type=int, default=5,
                    help="Number of Runs")
parser.add_argument('--arch', type=str, default='all', choices=model_names, nargs='+',
                    help='model architectures: ' + ' | '.join(model_names) + ' (default: all)')
parser.add_argument('--log-dir', type=str, default='log',
                    help='the path on the file system to place the working log directory at')
parser.add_argument('--filename', type=str, default='perf_test',
                    help='name of the output file')
parser.add_argument('--data-dir', type=str, required=True,
                    help='path to imagenet dataset')


model_names = sorted(name for name in models.__dict__
                     if not (not (name.islower() and not name.startswith("__"))
                             or not callable(models.__dict__[name])))


def get_env_pytorch_examples():
    pytorch_examples_home = os.environ.get('EXAMPLES_HOME')
    if pytorch_examples_home is None:
        print('EXAMPLES_HOME not found')
        sys.exit()

    return pytorch_examples_home


def execution(cmd, log_path):
    gc.collect()

    # logging
    log_file = open(log_path, "w+")
    log_file.write(cmd)
    log_file.write('\n')

    exec_command = shlex.split(cmd)
    proc = subprocess.Popen(exec_command, stdout=log_file, stderr=subprocess.STDOUT)
    proc.wait()
    return_code = proc.returncode
    log_file.close()

    log_file = open(log_path, 'r+')

    if return_code == 0:
        acc = parse_accuracy(log_file)
    else:
        acc = ('NA', 'NA')

    return acc


def parse_accuracy(log_file):
    output_data = log_file.readlines()
    _, _, prec1, _, prec2 = output_data[-2].split()
    return (prec1, prec2)


def config_runs(model, no_iter):
    iters = [i for i in range(no_iter)]
    if model == 'all':
        model = model_names

    return list(itertools.product(model, iters))


def cmd_string(examples_home, model, data_path):
    lr = 0.1
    if model in ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13_bn',
                 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
        lr = 0.01

    cmd = ' '.join(['python3', examples_home, '-a', model, '--lr', str(lr), data_path])
    return cmd


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
    temp_dir = join(folder_root, args.log_dir)

    log_init()

    time_now = str(datetime.now())
    logging.info('New performance test started at {}'.format(time_now))
    logging.info('model, prec1, prec5, iteration')

    examples_home = get_env_pytorch_examples()
    imagenet = join(examples_home, 'imagenet', 'main.py')

    runs = config_runs(args.arch, args.repeat)

    for i in tqdm(range(len(runs))):
        model, current_iter = runs[i]

        # logging execution
        file_name = runs[i][0] + '_' + str(runs[i][1]) + '.txt'
        log_path = join(temp_dir, file_name)

        # execution
        cmd = cmd_string(imagenet, model, args.data_dir)
        prec1, prec5 = execution(cmd, log_path)

        logging.info('{},{},{},{}'.format(model, prec1, prec5, current_iter))


if __name__ == '__main__':
    main()
