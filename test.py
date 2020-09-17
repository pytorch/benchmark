"""test.py
Setup and Run hub models.

Make sure to enable an https proxy if necessary, or the setup steps may hang.
"""
# This file shows how to use the benchmark suite from user end.
import argparse
import time
from torchbenchmark import workdir, setup, list_models
from unittest import TestCase
import re, sys, unittest
import os.path
import torch
import gc
import pprint

class TestBenchmark(TestCase):
    def setUp(self):
        if 'cuda' in str(self):
            self.memory = torch.cuda.memory_allocated()

    def tearDown(self):
        if 'cuda' in str(self):
            memory = torch.cuda.memory_allocated()
            n = gc.collect()
            print('Unreachable objects:', n)
            print('Remaining Garbage:', pprint.pprint(gc.garbage))
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                        del obj
                except: pass
            torch.cuda.empty_cache()
            self.assertEqual(self.memory, memory)

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
        m = model_object(self)
        try:
            module, example_inputs = m.get_module()
            module(*example_inputs)
        except NotImplementedError:
            self.skipTest('Method get_module is not implemented, skipping...')

    def train(self):
        m = model_object(self)
        try:
            start = time.time()
            m.train()
            print('Finished training on device: {} in {}s.'.format(device, time.time() - start))
        except NotImplementedError:
            self.skipTest('Method train is not implemented, skipping...')

    def eval(self):
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

_load_tests()
if __name__ == '__main__':
    unittest.main()
