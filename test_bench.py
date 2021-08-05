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
from torchbenchmark import list_models, _list_model_paths, ModelTask
from torchbenchmark.util.machine_config import get_machine_state
from torchbenchmark.util.model import no_grad

def pytest_generate_tests(metafunc):
    # This is where the list of models to test can be configured
    # e.g. by using info in metafunc.config
    devices = ['cpu', 'cuda']
    if metafunc.config.option.cpu_only:
        devices = ['cpu']

    repeats = 5

    if metafunc.cls and metafunc.cls.__name__ == "TestBenchNetwork":
        all_models = list_models()
        is_eval = metafunc.function.__name__ == "test_eval"
        test_name = lambda m : m.name + ("-freeze" if is_eval and hasattr(m, "optimized_for_inference") else "")
        metafunc.parametrize('model_class', all_models,
            ids=[test_name(m) for m in all_models], scope="class")

        metafunc.parametrize('device', devices, scope='class')
        metafunc.parametrize('compiler', ['jit', 'eager'], scope='class')

        # @pytest.mark.repeat(...) doesn't seem to work with benchmarks,
        # so this is the poor man's version.
        metafunc.parametrize('repeat_suffix', [f'{i}' for i in range(repeats)], scope='class')

    if metafunc.cls and metafunc.cls.__name__ == "TestBenchNetwork_Subprocess":
        # NB:
        #   It is expensive to do a clean load, so for the prototype we simply
        #   register based on the paths. TODO: consider using a threadpool to
        #   subprocess in parallel.
        model_paths = _list_model_paths()
        metafunc.parametrize('model_path', model_paths,
            ids=[os.path.basename(m) for m in model_paths], scope="class")
        metafunc.parametrize('device', devices, scope='class')
        metafunc.parametrize('compiler', ['jit', 'eager'], scope='class')

        # @pytest.mark.repeat(...) doesn't seem to work with benchmarks,
        # so this is the poor man's version.
        metafunc.parametrize('repeat_suffix', [f'subproc-{i}' for i in range(repeats)], scope='class')

@pytest.fixture(scope='function')
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

    def test_train(self, hub_model, repeat_suffix, benchmark):
        try:
            hub_model.set_train()
            benchmark(hub_model.train)
            benchmark.extra_info['machine_state'] = get_machine_state()
        except NotImplementedError:
            print('Method train is not implemented, skipping...')

    def test_eval(self, hub_model, repeat_suffix, benchmark, pytestconfig):
        try:
            ng_flag = hub_model.eval_in_nograd() and not pytestconfig.getoption("disable_nograd")
            with no_grad(ng_flag):
                hub_model.set_eval()
                benchmark(hub_model.eval)
                benchmark.extra_info['machine_state'] = get_machine_state()
                if pytestconfig.getoption("check_opt_vs_noopt_jit"):
                    hub_model.check_opt_vs_noopt_jit()
        except NotImplementedError:
            print('Method eval is not implemented, skipping...')


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=False,
    timer=time.perf_counter,
    group='hub',
)
class TestBenchNetwork_Subprocess:

    def test_train(self, model_path, device, compiler, repeat_suffix, benchmark):
        try:
            task = ModelTask(model_path)
            task.make_model_instance(device=device, jit=(compiler == 'jit'))
            task.set_train()
            benchmark(task.train)
            benchmark.extra_info['machine_state'] = get_machine_state()
        except:
            # TODO: fine grained Exception handling.
            print('Method train is not implemented, skipping...')
