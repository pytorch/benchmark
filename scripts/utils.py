import os
import gc
import itertools
import sys
import shlex
import subprocess
import torchvision.models as models

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
    cmd = ' '.join(['python3', examples_home, '-a', model, '--lr', str(lr),
                    '--epochs', str(1), data_path])
    return cmd
