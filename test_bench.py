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
from torchbenchmark import list_models, _list_model_paths, list_models_new, ModelTask
from torchbenchmark.util.machine_config import get_machine_state
from torchbenchmark.util.model import no_grad


def pytest_generate_tests(metafunc):
    # This is where the list of models to test can be configured
    # e.g. by using info in metafunc.config
    devices = ['cpu', 'cuda']
    if metafunc.config.option.cpu_only:
        devices = ['cpu']

    if metafunc.cls and metafunc.cls.__name__ == "TestBenchNetwork":
        models = [m for m in list_models_new() if m.exists]
        is_eval = metafunc.function.__name__ == "test_eval"
        test_name = lambda m : os.path.basename(m.path) + ("-freeze" if is_eval and m.optimized_for_inference else "")
        metafunc.parametrize(
            'model_path',
            [m.path for m in models],
            ids=[test_name(m) for m in models],
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
            task = ModelTask(model_path)
            task.make_model_instance(device=device, jit=(compiler == 'jit'))
            task.set_train()
            benchmark(task.train)
            benchmark.extra_info['machine_state'] = get_machine_state()

        except NotImplementedError:
            print('Method eval is not implemented, skipping...')

    def test_eval(self, model_path, device, compiler, benchmark, pytestconfig):
        try:
            task = ModelTask(model_path)
            task.make_model_instance(device=device, jit=(compiler == 'jit'))

            with task.no_grad(disable_nograd=pytestconfig.getoption("disable_nograd")):
                task.set_eval()
                benchmark(task.eval)
                benchmark.extra_info['machine_state'] = get_machine_state()
                if pytestconfig.getoption("check_opt_vs_noopt_jit"):
                    task.check_opt_vs_noopt_jit()

        except NotImplementedError:
            print('Method eval is not implemented, skipping...')
