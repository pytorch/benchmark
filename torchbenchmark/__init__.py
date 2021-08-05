import importlib
import os
import pathlib
import subprocess
import sys
from typing import Any, List, Optional, Tuple
from urllib import request

from torch.utils.benchmark._impl.tasks import base as base_task
from torch.utils.benchmark._impl.workers import subprocess_worker


proxy_suggestion = "Unable to verify https connectivity, " \
                   "required for setup.\n" \
                   "Do you need to use a proxy?"

this_dir = pathlib.Path(__file__).parent.absolute()
model_dir = 'models'
install_file = 'install.py'


def _test_https(test_url: str = 'https://github.com', timeout: float = 0.5) -> bool:
    try:
        request.urlopen(test_url, timeout=timeout)
    except OSError:
        return False
    return True


def _install_deps(model_path: str, verbose: bool = True) -> Tuple[bool, Any]:
    run_args = [
        [sys.executable, install_file],
    ]
    run_kwargs = {
        'cwd': model_path,
        'check': True,
    }
    try:
        if os.path.exists(os.path.join(model_path, install_file)):
            if not verbose:
                run_kwargs['stderr'] = subprocess.STDOUT
                run_kwargs['stdout'] = subprocess.PIPE
            subprocess.run(*run_args, **run_kwargs)  # type: ignore
        else:
            return (False, f"No install.py is found in {model_path}.")
    except subprocess.CalledProcessError as e:
        return (False, e.output)
    except Exception as e:
        return (False, e)

    return (True, None)


def _list_model_paths() -> List[str]:
    p = pathlib.Path(__file__).parent.joinpath(model_dir)
    return sorted(str(child.absolute()) for child in p.iterdir() if child.is_dir())


def setup(verbose: bool = True, continue_on_fail: bool = False) -> bool:
    if not _test_https():
        print(proxy_suggestion)
        sys.exit(-1)

    failures = {}
    for model_path in _list_model_paths():
        print(f"running setup for {model_path}...", end="", flush=True)
        success, errmsg = _install_deps(model_path, verbose=verbose)
        if success:
            print("OK")
        else:
            print("FAIL")
            try:
                errmsg = errmsg.decode()
            except Exception:
                pass
            failures[model_path] = errmsg
            if not continue_on_fail:
                break
    if verbose and len(failures):
        for model_path in failures:
            print(f"Error for {model_path}:")
            print("---------------------------------------------------------------------------")
            print(failures[model_path])
            print("---------------------------------------------------------------------------")
            print()

    return len(failures) == 0


class ModelTask(base_task.TaskBase):

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._worker = subprocess_worker.SubprocessWorker()
        self.worker.run("import torch")

        model_import_msg = self.maybe_import_model(
            package=__name__, model_path=model_path)

        if model_import_msg is None:
            self._model_present: bool = True
        else:
            assert isinstance(model_import_msg, str)
            print(model_import_msg)
            self._model_present = False

    @property
    def worker(self) -> subprocess_worker.SubprocessWorker:
        return self._worker

    @property
    def model_present(self) -> bool:
        return self._model_present

    def print_var(self, name: str) -> None:
        # TODO: maybe upstream?
        self.worker.run(f"__var_repr = repr({name})")
        print(self.worker.load("__var_repr"))
        self.worker.run("del __var_repr")

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def maybe_import_model(package: str, model_path: str) -> Optional[str]:
        import importlib
        import os

        model_name = os.path.basename(model_path)
        try:
            module = importlib.import_module(f'.models.{model_name}', package=package)

        except ModuleNotFoundError as e:
            return f"Warning: Could not find dependent module {e.name} for Model {model_name}, skip it"

        Model = getattr(module, 'Model', None)
        if Model is None:
            return f"Warning: {module} does not define attribute Model, skip it"

        if not hasattr(Model, 'name'):
            Model.name = model_name

        globals()["Model"] = Model

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def make_model_instance(device: str, jit: bool) -> None:
        Model = globals()["Model"]
        model = Model(device=device, jit=jit)

        import gc
        gc.collect()

        if device == 'cuda':
            torch.cuda.empty_cache()

        globals()["model"] = model

    def set_train(self) -> None:
        self.worker.run("model.set_train()")

    def train(self) -> None:
        self.worker.run("model.train()")


def list_models():
    models = []
    for model_path in _list_model_paths():
        model_name = os.path.basename(model_path)
        try:
            module = importlib.import_module(f'.models.{model_name}', package=__name__)
        except ModuleNotFoundError as e:
            print(f"Warning: Could not find dependent module {e.name} for Model {model_name}, skip it")
            continue
        Model = getattr(module, 'Model', None)
        if Model is None:
            print(f"Warning: {module} does not define attribute Model, skip it")
            continue
        if not hasattr(Model, 'name'):
            Model.name = model_name
        models.append(Model)
    return models
