"""test.py
Setup and Run hub models.

Make sure to enable an https proxy if necessary, or the setup steps may hang.
"""

# This file shows how to use the benchmark suite from user end.
import gc
import os
import unittest

import torch
from torchbenchmark import (
    _list_canary_model_paths,
    _list_model_paths,
    get_metadata_from_yaml,
    ModelTask,
)
from torchbenchmark.util.metadata_utils import skip_by_metadata


# Some of the models have very heavyweight setup, so we have to set a very
# generous limit. That said, we don't want the entire test suite to hang if
# a single test encounters an extreme failure, so we give up after a test is
# unresponsive to 5 minutes by default. (Note: this does not require that the
# entire test case completes in 5 minutes. It requires that if the worker is
# unresponsive for 5 minutes the parent will presume it dead / incapacitated.)
TIMEOUT = int(os.getenv("TIMEOUT", 300))  # Seconds


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        gc.collect()

    def tearDown(self):
        gc.collect()


def _create_example_model_instance(task: ModelTask, device: str, mode: str):
    skip = False
    extra_args = ["--accuracy"]
    if mode == "inductor":
        extra_args.append("--inductor")
    try:
        task.make_model_instance(test="eval", device=device, extra_args=extra_args)
    except NotImplementedError:
        try:
            task.make_model_instance(
                test="train", device=device, extra_args=extra_args
            )
        except NotImplementedError:
            skip = True
    finally:
        if skip:
            raise NotImplementedError(
                f"Model is not implemented on the device {device}"
            )


def _load_test(path, device, mode):
    model_name = os.path.basename(path)

    def _skip_cuda_memory_check_p(metadata):
        if device != "cuda":
            return True
        if "skip_cuda_memory_leak" in metadata and metadata["skip_cuda_memory_leak"]:
            return True
        return False

    def example_fn(self):
        task = ModelTask(model_name, timeout=TIMEOUT)
        with task.watch_cuda_memory(
            skip=_skip_cuda_memory_check_p(metadata), assert_equal=self.assertEqual
        ):
            try:
                _create_example_model_instance(task, device, mode)
                accuracy = task.get_model_attribute("accuracy")
                assert (
                    accuracy == "pass"
                    or accuracy == "eager_1st_run_OOM"
                    or accuracy == "eager_2nd_run_OOM"
                ), f"Expected accuracy pass, get {accuracy}"
                task.del_model_instance()
            except NotImplementedError as e:
                self.skipTest(
                    f'Accuracy check on {device} is not implemented because "{e}", skipping...'
                )

    def train_fn(self):
        metadata = get_metadata_from_yaml(path)
        task = ModelTask(model_name, timeout=TIMEOUT)
        allow_customize_batch_size = task.get_model_attribute(
            "ALLOW_CUSTOMIZE_BSIZE", classattr=True
        )
        # to speedup test, use batch size 1 if possible
        batch_size = 1 if allow_customize_batch_size else None
        with task.watch_cuda_memory(
            skip=_skip_cuda_memory_check_p(metadata), assert_equal=self.assertEqual
        ):
            try:
                task.make_model_instance(
                    test="train", device=device, batch_size=batch_size, extra_args=["--inductor"] if mode == "inductor" else []
                )
                task.invoke()
                task.check_details_train(device=device, md=metadata)
                task.del_model_instance()
            except NotImplementedError as e:
                self.skipTest(
                    f'Method train on {device} is not implemented because "{e}", skipping...'
                )

    def eval_fn(self):
        metadata = get_metadata_from_yaml(path)
        task = ModelTask(model_name, timeout=TIMEOUT)
        allow_customize_batch_size = task.get_model_attribute(
            "ALLOW_CUSTOMIZE_BSIZE", classattr=True
        )
        # to speedup test, use batch size 1 if possible
        batch_size = 1 if allow_customize_batch_size else None
        with task.watch_cuda_memory(
            skip=_skip_cuda_memory_check_p(metadata), assert_equal=self.assertEqual
        ):
            try:
                task.make_model_instance(
                    test="eval", device=device, batch_size=batch_size, extra_args=["--inductor"] if mode == "inductor" else []
                )
                task.invoke()
                task.check_details_eval(device=device, md=metadata)
                task.check_eval_output()
                task.del_model_instance()
            except NotImplementedError as e:
                self.skipTest(
                    f'Method eval on {device} is not implemented because "{e}", skipping...'
                )

    def check_device_fn(self):
        task = ModelTask(model_name, timeout=TIMEOUT)
        with task.watch_cuda_memory(
            skip=_skip_cuda_memory_check_p(metadata), assert_equal=self.assertEqual
        ):
            try:
                task.make_model_instance(test="eval", device=device, extra_args=["--inductor"] if mode == "inductor" else [])
                task.check_device()
                task.del_model_instance()
            except NotImplementedError as e:
                self.skipTest(
                    f'Method check_device on {device} is not implemented because "{e}", skipping...'
                )

    metadata = get_metadata_from_yaml(path)
    for fn, fn_name in zip(
        [example_fn, train_fn, eval_fn, check_device_fn],
        ["example", "train", "eval", "check_device"],
    ):
        # set exclude list based on metadata
        setattr(
            TestBenchmark,
            f"test_{model_name}_{fn_name}_{device}_{mode}",
            (
                unittest.skipIf(
                    # This is expecting that models will never be skipped just based on backend, just on eval or train functions being implemented
                    skip_by_metadata(
                        test=fn_name, device=device, extra_args=[], metadata=metadata
                    ),
                    "This test is skipped by its metadata",
                )(fn)
            ),
        )


def _load_tests():
    modes = ["eager", "inductor"]
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    if device := os.getenv("ACCELERATOR"):
        devices.append(device)
    model_paths = _list_model_paths()
    if os.getenv("USE_CANARY_MODELS"):
        model_paths.extend(_list_canary_model_paths())
    for path in model_paths:
        # TODO: skipping quantized tests for now due to BC-breaking changes for prepare
        # api, enable after PyTorch 1.13 release
        if "quantized" in path:
            continue
        for device in devices:
            for mode in modes:
                _load_test(path, device, mode)


_load_tests()
if __name__ == "__main__":
    unittest.main()
