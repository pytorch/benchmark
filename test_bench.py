"""test_bench.py
Runs hub models in benchmark mode using pytest-benchmark. Run setup separately first.

Usage:
  python test.py --setup_only
  pytest test_bench.py

See pytest-benchmark help (pytest test_bench.py -h) for additional options
e.g. --benchmark-autosave
     --benchmark-compare
     -k <filter expression>
     ...
"""
import os
import gc
import pytest
import time
import torch
from components._impl.workers import subprocess_worker
from torchbenchmark import _list_model_paths, ModelTask
from torchbenchmark.util.machine_config import get_machine_state

# Some of the models have very heavyweight setup, so we have to set a very
# generous limit. That said, we don't want the entire test suite to hang if
# a single test encounters an extreme failure, so we give up after a test
# is unresponsive for 20 minutes. (Note: this does not require that the entire
# test case completes in 20 minutes. It requires that if the worker is
# unresponsive for 20 minutes the parent will presume it dead / incapacitated.)
TIMEOUT = 1200  # Seconds

def pytest_generate_tests(metafunc):
    # This is where the list of models to test can be configured
    # e.g. by using info in metafunc.config
    devices = ['cpu', 'cuda']
    if metafunc.config.option.cpu_only:
        devices = ['cpu']

    if metafunc.cls and metafunc.cls.__name__ == "TestBenchNetwork":
        paths = _list_model_paths()
        metafunc.parametrize(
            'model_path', paths,
            ids=[os.path.basename(path) for path in paths],
            scope="class")

        metafunc.parametrize('device', devices, scope='class')
        metafunc.parametrize('compiler', ['jit', 'eager'], scope='class')


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=False,
    timer=time.perf_counter,
    group='hub',
)
class TestBenchNetwork:

    def test_train(self, model_path, device, compiler, benchmark):
        try:
            task = ModelTask(model_path, timeout=TIMEOUT)
            if not task.model_details.exists:
                return  # Model is not supported.

            task.make_model_instance(device=device, jit=(compiler == 'jit'))
            task.set_train()
            benchmark(task.train)
            benchmark.extra_info['machine_state'] = get_machine_state()

        except NotImplementedError:
            print('Method eval is not implemented, skipping...')

    def test_eval(self, model_path, device, compiler, benchmark, pytestconfig):
        try:
            task = ModelTask(model_path, timeout=TIMEOUT)
            if not task.model_details.exists:
                return  # Model is not supported.

            task.make_model_instance(device=device, jit=(compiler == 'jit'))

            with task.no_grad(disable_nograd=pytestconfig.getoption("disable_nograd")):
                task.set_eval()
                benchmark(task.eval)
                benchmark.extra_info['machine_state'] = get_machine_state()
                if pytestconfig.getoption("check_opt_vs_noopt_jit"):
                    task.check_opt_vs_noopt_jit()

        except NotImplementedError:
            print('Method eval is not implemented, skipping...')


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=False,
    timer=time.perf_counter,
    group='hub',
)
class TestWorker:
    """Benchmark SubprocessWorker to make sure we aren't skewing results."""

    def test_worker_noop(self, benchmark):
        worker = subprocess_worker.SubprocessWorker(timeout=TIMEOUT)
        benchmark(lambda: worker.run("pass"))

    def test_worker_store(self, benchmark):
        worker = subprocess_worker.SubprocessWorker(timeout=TIMEOUT)
        benchmark(lambda: worker.store("x", 1))

    def test_worker_load(self, benchmark):
        worker = subprocess_worker.SubprocessWorker(timeout=TIMEOUT)
        worker.store("x", 1)
        benchmark(lambda: worker.load("x"))
