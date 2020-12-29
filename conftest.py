import os
import pytest
import torch
import util.machine_config as mc

def pytest_addoption(parser):
    parser.addoption("--fuser", help="fuser to use for benchmarks")
    parser.addoption("--ignore_machine_config", action="store_true", help="Disable check for machine being configured for stable benchmarking.")

@pytest.fixture(scope="session", autouse=True)
def check_machine_configured(request):
    if not request.config.getoption('ignore_machine_config'):
        mc.check_machine_configured()

def set_fuser(fuser):
    if fuser == "old":
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser == "te":
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_bailout_depth(20)
        torch._C._jit_set_num_profiled_runs(2)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)

def pytest_configure(config):
    set_fuser(config.getoption("fuser"))

def pytest_benchmark_update_machine_info(config, machine_info):
    machine_info['pytorch_version'] = torch.__version__
    try:
        import torchtext
        machine_info['torchtext_version'] = torchtext.__version__
    except ImportError:
        machine_info['torchtext_version'] = '*not-installed*'

    try:
        import torchvision
        machine_info['torchvision_version'] = torchvision.__version__
    except ImportError:
        machine_info['torchvision_version'] = '*not-installed*'

    machine_info['circle_build_num'] = os.environ.get("CIRCLE_BUILD_NUM")
    machine_info['circle_project_name'] = os.environ.get("CIRCLE_PROJECT_REPONAME")
    machine_info['torchbench_machine_config'] = mc.get_machine_config()
    machine_info['torchbench_machine_state'] = mc.get_machine_state()