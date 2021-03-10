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
from torchbenchmark import list_models
from torchbenchmark.util.machine_config import get_machine_state
from torchbenchmark.util.model import no_grad

def pytest_generate_tests(metafunc):
    # This is where the list of models to test can be configured
    # e.g. by using info in metafunc.config
    all_models = list_models()
    if metafunc.cls and metafunc.cls.__name__ == "TestBenchNetwork":
        metafunc.parametrize('model_class', all_models,
                             ids=[m.name for m in all_models], scope="class")
        metafunc.parametrize('device', ['cpu', 'cuda'], scope='class')
        metafunc.parametrize('compiler', ['jit', 'eager'], scope='class')

@pytest.fixture(scope='class')
def hub_model(request, model_class, device, compiler):
    """Constructs a model object for pytests to use.
    Any pytest function that consumes a 'modeldef' arg will invoke this
    automatically, and reuse it for each test that takes that combination
    of arguments within the module.

    If reusing the module between tests isn't safe, change 'scope' parameter.
    """
    use_jit = compiler == 'jit'
    return model_class(device=device, jit=use_jit)

    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


def cuda_timer():
    torch.cuda.synchronize()
    return time.perf_counter()


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=True,
    timer=cuda_timer if (torch.has_cuda and
                         torch.cuda.is_available()) else time.perf_counter,
    group='hub',
)
class TestBenchNetwork:
    """
    This test class will get instantiated once for each 'model_stuff' provided
    by the fixture above, for each device listed in the device parameter.
    """
    def test_train(self, hub_model, benchmark):
        try:
            hub_model.set_train()
            benchmark(hub_model.train)
            benchmark.extra_info['machine_state'] = get_machine_state()
        except NotImplementedError:
            print('Method train is not implemented, skipping...')

    def test_eval(self, hub_model, benchmark, pytestconfig):
        try:
            ng_flag = hub_model.eval_in_nograd() and not pytestconfig.getoption("disable_nograd")
            with no_grad(ng_flag):
                hub_model.set_eval_and_freeze()
                benchmark(hub_model.eval)
                benchmark.extra_info['machine_state'] = get_machine_state()
                if pytestconfig.getoption("check_opt_vs_noopt_jit"):
                    hub_model.check_opt_vs_noopt_jit()
        except NotImplementedError:
            print('Method eval is not implemented, skipping...')
