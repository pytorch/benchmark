import contextlib
import dataclasses
import importlib
import multiprocessing
import multiprocessing.dummy
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
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


@dataclasses.dataclass(frozen=True)
class ModelDetails:
    path: str
    exists: bool
    optimized_for_inference: bool
    _diagnostic_msg: str


class ModelTask(base_task.TaskBase):

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._worker = subprocess_worker.SubprocessWorker()
        self.worker.run("import torch")

        self._details: ModelDetails = ModelDetails(
            **self._maybe_import_model(
                package=__name__,
                model_path=model_path,
            )
        )

        if self._details._diagnostic_msg:
            print(self._details._diagnostic_msg)

    @property
    def worker(self) -> subprocess_worker.SubprocessWorker:
        return self._worker

    @property
    def model_details(self) -> bool:
        return self._details

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def _maybe_import_model(package: str, model_path: str) -> Dict[str, Any]:
        import importlib
        import os

        model_name = os.path.basename(model_path)
        diagnostic_msg = ""
        try:
            module = importlib.import_module(f'.models.{model_name}', package=package)
            Model = getattr(module, 'Model', None)
            if Model is None:
                diagnostic_msg = f"Warning: {module} does not define attribute Model, skip it"

            elif not hasattr(Model, 'name'):
                Model.name = model_name

        except ModuleNotFoundError as e:
            Model = None
            diagnostic_msg = f"Warning: Could not find dependent module {e.name} for Model {model_name}, skip it"

        globals()["Model"] = Model

        return {
            "path": model_path,
            "exists": Model is not None,
            "optimized_for_inference": hasattr(Model, "optimized_for_inference"),
            "_diagnostic_msg": diagnostic_msg,
        }

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def make_model_instance(device: str, jit: bool) -> None:
        Model = globals()["Model"]
        model = Model(device=device, jit=jit)

        import gc
        gc.collect()

        if device == 'cuda':
            torch.cuda.empty_cache()
            maybe_sync = torch.cuda.synchronize
        else:
            maybe_sync = lambda: None

        globals().update({
            "model": model,
            "maybe_sync": maybe_sync,
        })

    def set_train(self) -> None:
        self.worker.run("model.set_train()")

    def train(self) -> None:
        self.worker.run("""
            model.train()
            maybe_sync()
        """)

    @contextlib.contextmanager
    def no_grad(self, disable_nograd: bool):
        # TODO: deduplicate with `torchbenchmark.util.model.no_grad`

        initial_value = self.worker.load_stmt("torch.is_grad_enabled()")
        eval_in_nograd = (
            not disable_nograd and
            self.worker.load_stmt("model.eval_in_nograd()")
        )

        try:
            self.worker.run(f"torch.set_grad_enabled({not eval_in_nograd})")
            yield
        finally:
            self.worker.run(f"torch.set_grad_enabled({initial_value})")

    def set_eval(self) -> None:
        self.worker.run("model.set_eval()")

    def eval(self) -> None:
        self.worker.run("""
            model.eval()
            maybe_sync()
        """)

    def check_opt_vs_noopt_jit(self) -> None:
        self.worker.run("model.check_opt_vs_noopt_jit()")


def list_models_details(workers: int=1) -> List[ModelDetails]:
    # A lot of the work of importing the models to check is single threaded,
    # so we can save a lot of headache by using multiple workers. However it's
    # not linear, and past about cpu_count / 2 the returns are marginal.
    num_workers = max(1, int(multiprocessing.cpu_count() // 2))
    with multiprocessing.dummy.Pool(num_workers) as pool:
        return pool.map(
            lambda model_path: ModelTask(model_path).model_details,
            _list_model_paths()
        )


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
