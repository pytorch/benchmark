import os
import gc
import itertools
import sys
import shlex
import subprocess
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if not (not (name.islower() and not name.startswith("__")) or not callable(models.__dict__[name])))


def get_env_pytorch_examples():
    pytorch_examples_home = os.environ.get('PYTORCH_EXAMPLES_HOME')
    if pytorch_examples_home is None:
        print('PYTORCH_EXAMPLES_HOME not found')
        sys.exit()

    return pytorch_examples_home


def execution(cmd, log_path):
    gc.collect()

    # log all executions
    log_file = open(log_path, "w+")
    log_file.write(cmd)
    log_file.write('\n')

    exec_command = shlex.split(cmd)
    #proc = subprocess.Popen(exec_command, stdout=log_file, stderr=subprocess.STDOUT)
    #proc.wait()
    #return_code = proc.returncode
    return_code = 1

    if return_code == 0:
        acc = '0.97'
    else:
        acc = 'error'
    return acc


def config_runs(model, iter):
    iters = [i for i in range(iter)]
    if model == 'all':
        models = model_names

    return list(itertools.product(models, iters))


def cmd_string(examples_home, model, data_path):
    lr = 0.1
    if model in ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
        lr = 0.01
    cmd = ' '.join(['python3', examples_home, '-a', model, '--lr', str(lr), data_path])
    return cmd
