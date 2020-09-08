"""test.py
Setup and Run hub models.

Make sure to enable an https proxy if necessary, or the setup steps may hang.
"""
# This file shows how to use the benchmark suite from user end.
import argparse
import time
from bench_utils import workdir, setup, list_models
from unittest import TestCase
import re, sys, unittest
import os.path
import torch

class TestBenchmark(TestCase):
    pass

def run_model(model_class, model_path, device):
    m = model_class(device=device)


def _load_test(model_class, model_path, device):
    dir, name = os.path.split(model_path)
    name = re.sub('[^A-Za-z0-9_]+', '_', name)

    def model_object(self):
        if device == 'cuda' and not torch.cuda.is_available():
            self.skipTest("torch.cuda not available")
        return model_class(device=device)

    def example(self):
        with workdir(model_path):
            m = model_object(self)
            try:
                module, example_inputs = m.get_module()
                module(*example_inputs)
            except NotImplementedError:
                self.skipTest('Method get_module is not implemented, skipping...')

    def train(self):
        with workdir(model_path):
            m = model_object(self)
            try:
                start = time.time()
                m.train()
                print('Finished training on device: {} in {}s.'.format(device, time.time() - start))
            except NotImplementedError:
                self.skipTest('Method train is not implemented, skipping...')

    def eval(self):
        with workdir(model_path):
            m = model_object(self)
            try:
                start = time.time()
                m.eval()
                print('Finished eval on device: {} in {}s.'.format(device, time.time() - start))
            except NotImplementedError:
                self.skipTest('Method eval is not implemented, skipping...')

    setattr(TestBenchmark, f'test_{name}_example_{device}', example)
    setattr(TestBenchmark, f'test_{name}_train_{device}', train)
    setattr(TestBenchmark, f'test_{name}_eval_{device}', eval)


def _load_tests():
    for model, model_path in list_models():
        for device in ('cpu', 'cuda'):
            _load_test(model, model_path, device)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup_only', action='store_true',
                        help='run setup steps only, then exit.')
    parser.add_argument('--no_setup', action='store_true',
                        help='skip the setup process.')

    args, unknown = parser.parse_known_args()

    if not args.no_setup:
        setup()
    if not args.setup_only:
        _load_tests()
        unittest.main(argv=[sys.argv[0]] + unknown)
else:
    # being run as part of a test suite, assume setup has already been run separately
    _load_tests()