"""test.py
Setup and Run hub models.

Make sure to enable an https proxy if necessary, or the setup steps may hang.
"""
# This file shows how to use the benchmark suite from user end.
import gc
import functools
import os
import traceback
import unittest
from unittest.mock import patch

import torch
from torchbenchmark import _list_model_paths, ModelTask


# Some of the models have very heavyweight setup, so we have to set a very
# generous limit. That said, we don't want the entire test suite to hang if
# a single test encounters an extreme failure, so we give up after 5 a test
# is unresponsive to 5 minutes. (Note: this does not require that the entire
# test case completes in 5 minutes. It requires that if the worker is
# unresponsive for 5 minutes the parent will presume it dead / incapacitated.)
TIMEOUT = 300  # Seconds


class TestBenchmark(unittest.TestCase):

    def setUp(self):
        gc.collect()

    def tearDown(self):
        gc.collect()

    def test_fx_profile(self):
        try:
            from torch.fx.interpreter import Interpreter
        except ImportError:  # older versions of PyTorch
            raise unittest.SkipTest("Requires torch>=1.8")
        from fx_profile import main, ProfileAggregate
        with patch.object(ProfileAggregate, "save") as mock_save:
            # just run one model to make sure things aren't completely broken
            main(["--repeat=1", "--filter=pytorch_struct", "--device=cpu"])
            self.assertGreaterEqual(mock_save.call_count, 1)


def _load_test(path, device):

    def example(self):
        task = ModelTask(path, timeout=TIMEOUT)
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                task.make_model_instance(device=device, jit=False)
                task.check_example()
                task.del_model_instance()

            except NotImplementedError:
                self.skipTest('Method get_module is not implemented, skipping...')

    def train(self):
        task = ModelTask(path, timeout=TIMEOUT)
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                task.make_model_instance(device=device, jit=False)
                task.set_train()
                task.train()
                task.del_model_instance()
            except NotImplementedError:
                self.skipTest('Method train is not implemented, skipping...')

    def eval_fn(self):
        task = ModelTask(path, timeout=TIMEOUT)
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                task.make_model_instance(device=device, jit=False)
                assert (
                    not task.model_details.optimized_for_inference or
                    task.worker.load_stmt("hasattr(model, 'eval_model')"))

                task.set_eval()
                task.eval()
                task.del_model_instance()
            except NotImplementedError:
                self.skipTest('Method eval is not implemented, skipping...')

    name = os.path.basename(path)
    setattr(TestBenchmark, f'test_{name}_example_{device}', example)
    setattr(TestBenchmark, f'test_{name}_train_{device}', train)
    setattr(TestBenchmark, f'test_{name}_eval_{device}', eval_fn)


def _load_tests():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for path in _list_model_paths():
        for device in devices:
            _load_test(path, device)


_load_tests()
if __name__ == '__main__':
    unittest.main()
