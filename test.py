"""test.py
Setup and Run hub models.

Make sure to enable an https proxy if necessary, or the setup steps may hang.
"""
# This file shows how to use the benchmark suite from user end.
import gc
import unittest
from unittest import TestCase
from unittest.mock import patch

import torch
from torchbenchmark import list_models_details, ModelTask


class TestBenchmark(TestCase):

    # We can't use setUp and tearDown, because the task (and by extension the
    # worker) is created on a per-test basis.
    def make_task(self, path, device):
        task = ModelTask(path)
        task.worker.run("import gc;gc.collect()")
        if device == 'cuda':
            self.memory = task.worker.load_stmt("torch.cuda.memory_allocated()")
        task.make_model_instance(device=device, jit=False)
        return task

    def assert_cleanup(self, task, device):
        if device == "cuda":
            task.worker.run("""
                del model
                gc.collect()
            """)
            memory = task.worker.load_stmt("torch.cuda.memory_allocated()")
            task.worker.run("torch.cuda.empty_cache()")
            self.assertEqual(self.memory, memory)

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


def _load_test(details, device):

    def example(self):
        task = self.make_task(details.path, device)

        try:
            task.worker.run("""
                module, example_inputs = model.get_module()
                if isinstance(example_inputs, dict):
                    # Huggingface models pass **kwargs as arguments, not *args
                    module(**example_inputs)
                else:
                    module(*example_inputs)
                del module
                del example_inputs
            """)
            self.assert_cleanup(task, device)
        except NotImplementedError:
            self.skipTest('Method get_module is not implemented, skipping...')

    def train(self):
        task = self.make_task(details.path, device)

        try:
            task.set_train()
            task.train()
            self.assert_cleanup(task, device)
        except NotImplementedError:
            self.skipTest('Method train is not implemented, skipping...')

    def eval_fn(self):
        task = self.make_task(details.path, device)
        if details.optimized_for_inference:
            assert task.worker.load_stmt("hasattr(model, 'eval_model')")

        try:
            task.set_eval()
            task.eval()
            self.assert_cleanup(task, device)
        except NotImplementedError:
            self.skipTest('Method eval is not implemented, skipping...')

    setattr(TestBenchmark, f'test_{details.name}_example_{device}', example)
    setattr(TestBenchmark, f'test_{details.name}_train_{device}', train)
    setattr(TestBenchmark, f'test_{details.name}_eval_{device}', eval_fn)


def _load_tests():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for details in list_models_details():
        for device in devices:
            _load_test(details, device)


_load_tests()
if __name__ == '__main__':
    unittest.main()
