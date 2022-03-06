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
import yaml

import torch
from torchbenchmark import _list_model_paths, ModelTask, get_metadata_from_yaml


# Some of the models have very heavyweight setup, so we have to set a very
# generous limit. That said, we don't want the entire test suite to hang if
# a single test encounters an extreme failure, so we give up after 5 a test
# is unresponsive to 5 minutes. (Note: this does not require that the entire
# test case completes in 5 minutes. It requires that if the worker is
# unresponsive for 5 minutes the parent will presume it dead / incapacitated.)
TIMEOUT = 300  # Seconds

# Skip this list of unit tests. One reason may be that the original batch size
# used in the paper is too large to fit on the CI's GPU.
EXCLUDELIST = {("densenet121", "train", "cuda"),  # GPU train runs out of memory on CI.
               ("densenet121", "train", "cpu"),  # CPU train runs for too long on CI.
               ("densenet121", "example", "cuda"),  # GPU train runs out of memory on CI.
               ("densenet121", "example", "cpu")}  # CPU train runs for too long on CI.


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

    def example_fn(self):
        task = ModelTask(path, timeout=TIMEOUT)
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                task.make_model_instance(test="eval", device=device, jit=False)
                task.check_example()
                task.del_model_instance()

            except NotImplementedError:
                self.skipTest('Method get_module is not implemented, skipping...')

    def train_fn(self):
        metadata = get_metadata_from_yaml(path)
        task = ModelTask(path, timeout=TIMEOUT)
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                task.make_model_instance(test="train", device=device, jit=False)
                task.set_train()
                task.train()
                task.check_details_train(device=device, md=metadata)
                task.del_model_instance()
            except NotImplementedError:
                self.skipTest(f'Method train on {device} is not implemented, skipping...')

    def eval_fn(self):
        metadata = get_metadata_from_yaml(path)
        task = ModelTask(path, timeout=TIMEOUT)
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                task.make_model_instance(test="eval", device=device, jit=False)

                task.set_eval()
                task.eval()
                task.check_details_eval(device=device, md=metadata)
                task.check_eval_output()
                task.del_model_instance()
            except NotImplementedError:
                self.skipTest(f'Method eval on {device} is not implemented, skipping...')

    def check_device_fn(self):
        task = ModelTask(path, timeout=TIMEOUT)
        with task.watch_cuda_memory(skip=(device != "cuda"), assert_equal=self.assertEqual):
            try:
                task.make_model_instance(test="eval", device=device, jit=False)
                task.check_device()
                task.del_model_instance()
            except NotImplementedError:
                self.skipTest(f'Method check_device on {device} is not implemented, skipping...')

    name = os.path.basename(path)
    for fn, fn_name in zip([example_fn, train_fn, eval_fn, check_device_fn],
                           ["example", "train", "eval", "check_device"]):
        setattr(TestBenchmark, f'test_{name}_{fn_name}_{device}',
                (unittest.skipIf((name, fn_name, device) in EXCLUDELIST, "This test is on the EXCLUDELIST")(fn)))


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
