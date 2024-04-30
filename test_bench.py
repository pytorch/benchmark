"""test_bench.py
Runs hub models in benchmark mode using pytest-benchmark. Run setup separately first.

Usage:
  python install.py
  pytest test_bench.py

See pytest-benchmark help (pytest test_bench.py -h) for additional options
e.g. --benchmark-autosave
     --benchmark-compare
     -k <filter expression>
     ...
"""

import os
import time

import pytest
import torch
from torchbenchmark import _list_model_paths, get_metadata_from_yaml, ModelTask
from torchbenchmark._components._impl.workers import subprocess_worker
from torchbenchmark.util.machine_config import get_machine_state
from torchbenchmark.util.metadata_utils import skip_by_metadata


def pytest_generate_tests(metafunc):
    # This is where the list of models to test can be configured
    # e.g. by using info in metafunc.config
    devices = ["cpu", "cuda"]

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    if metafunc.config.option.cpu_only:
        devices = ["cpu"]

    if metafunc.config.option.cuda_only:
        devices = ["cuda"]

    if metafunc.config.option.mps_only:
        devices = ["mps"]

    if metafunc.cls and metafunc.cls.__name__ == "TestBenchNetwork":
        paths = _list_model_paths()
        model_names = [os.path.basename(path) for path in paths]
        metafunc.parametrize(
            "model_name",
            model_names,
            ids=model_names,
            scope="class",
        )

        metafunc.parametrize("device", devices, scope="class")


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=False,
    timer=time.perf_counter,
    group="hub",
)
class TestBenchNetwork:

    def test_train(self, model_name, device, compiler, benchmark):
        try:
            if skip_by_metadata(
                test="train",
                device=device,
                extra_args=[],
                metadata=get_metadata_from_yaml(model_name),
            ):
                raise NotImplementedError("Test skipped by its metadata.")
            # TODO: skipping quantized tests for now due to BC-breaking changes for prepare
            # api, enable after PyTorch 1.13 release
            if "quantized" in model_name:
                return
            task = ModelTask(model_name)
            if not task.model_details.exists:
                return  # Model is not supported.

            task.make_model_instance(test="train", device=device)
            benchmark(task.invoke)
            benchmark.extra_info["machine_state"] = get_machine_state()
            benchmark.extra_info["batch_size"] = task.get_model_attribute("batch_size")
            benchmark.extra_info["precision"] = task.get_model_attribute(
                "dargs", "precision"
            )
            benchmark.extra_info["test"] = "train"

        except NotImplementedError:
            print(f"Test train on {device} is not implemented, skipping...")

    def test_eval(self, model_name, device, compiler, benchmark, pytestconfig):
        try:
            if skip_by_metadata(
                test="eval",
                device=device,
                extra_args=[],
                metadata=get_metadata_from_yaml(model_name),
            ):
                raise NotImplementedError("Test skipped by its metadata.")
            # TODO: skipping quantized tests for now due to BC-breaking changes for prepare
            # api, enable after PyTorch 1.13 release
            if "quantized" in model_name:
                return
            task = ModelTask(model_name)
            if not task.model_details.exists:
                return  # Model is not supported.

            task.make_model_instance(test="eval", device=device)

            with task.no_grad(disable_nograd=pytestconfig.getoption("disable_nograd")):
                benchmark(task.invoke)
                benchmark.extra_info["machine_state"] = get_machine_state()
                benchmark.extra_info["batch_size"] = task.get_model_attribute(
                    "batch_size"
                )
                benchmark.extra_info["precision"] = task.get_model_attribute(
                    "dargs", "precision"
                )
                benchmark.extra_info["test"] = "eval"

        except NotImplementedError:
            print(f"Test eval on {device} is not implemented, skipping...")


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=False,
    timer=time.perf_counter,
    group="hub",
)
class TestWorker:
    """Benchmark SubprocessWorker to make sure we aren't skewing results."""

    def test_worker_noop(self, benchmark):
        worker = subprocess_worker.SubprocessWorker()
        benchmark(lambda: worker.run("pass"))

    def test_worker_store(self, benchmark):
        worker = subprocess_worker.SubprocessWorker()
        benchmark(lambda: worker.store("x", 1))

    def test_worker_load(self, benchmark):
        worker = subprocess_worker.SubprocessWorker()
        worker.store("x", 1)
        benchmark(lambda: worker.load("x"))
