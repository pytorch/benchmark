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
import gc
import os
import pytest
import time
import torch
from bench_utils import workdir, list_model_paths


def pytest_generate_tests(metafunc, display_len=24):
    # This is where the list of models to test can be configured
    # e.g. by using info in metafunc.config
    all_models = list_model_paths()
    short_names = []
    for name in all_models:
        short = os.path.split(name)[1]
        if len(short) > display_len:
            short = short[:display_len] + "..."
        short_names.append(short)
    metafunc.parametrize('model_path', all_models,
                         ids=short_names, scope="module")
    metafunc.parametrize('device', ['cpu', 'cuda'], scope='module')
    metafunc.parametrize('compiler', ['jit', 'eager'], scope='module')


def get_cuda_mem_in_use():
    if torch.has_cuda and torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    else:
        return 0


def assert_cuda_mem(used, device):
    """Sanity check, it's easy to accidentally add a new model which purportedly
    supports cuda/cpu but silently runs on the other.
    """
    if device == 'cuda':
        assert used > 0, "Model allocated no gpu memory in cuda mode"
    else:
        assert used == 0, "Model configured in cpu mode used gpu memory"


@pytest.fixture(scope='class')
def model_cfg(request, model_path, device, compiler):
    """Constructs a model object for pytests to use.
    Any pytest function that consumes a 'modeldef' arg will invoke this
    automatically, and reuse it for each test that takes that combination
    of arguments within the module.
    """
    hubconf_file = 'hubconf.py'
    dbgname = os.path.split(model_path)[1]
    cuda_mem_initial = get_cuda_mem_in_use()
    with workdir(model_path):
        hub_module = torch.hub.import_module(hubconf_file, hubconf_file)
        Model = getattr(hub_module, 'Model', None)
        if not Model:
            raise RuntimeError('Missing class Model in {}/hubconf.py'
                               .format(model_path))
        use_jit = compiler == 'jit'
        m = {'model': Model(device=device, jit=use_jit)}
        yield m
        assert_cuda_mem(get_cuda_mem_in_use() - cuda_mem_initial, device)

        # The thing yielded from the fixture is kept alive until after all tests run, but
        # we want the model resources to be deallocated before the next model runs.
        del m['model']
        gc.collect()
        assert torch.cuda.memory_allocated() == 0


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
    def test_train(self, model_cfg, benchmark):
        try:
            benchmark(model_cfg['model'].train)
        except NotImplementedError:
            print('Method train is not implemented, skipping...')

    def test_eval(self, model_cfg, benchmark):
        try:
            benchmark(model_cfg['model'].eval)
        except NotImplementedError:
            print('Method eval is not implemented, skipping...')
