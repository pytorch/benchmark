import contextlib
import dataclasses
import importlib
import io
import multiprocessing
import multiprocessing.dummy
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple
from urllib import request

from components._impl.tasks import base as base_task
from components._impl.workers import subprocess_worker


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

    output_buffer = None
    _, stdout_fpath = tempfile.mkstemp()
    try:
        output_buffer = io.FileIO(stdout_fpath, mode="w")
        if os.path.exists(os.path.join(model_path, install_file)):
            if not verbose:
                run_kwargs['stderr'] = subprocess.STDOUT
                run_kwargs['stdout'] = output_buffer
            subprocess.run(*run_args, **run_kwargs)  # type: ignore
        else:
            return (False, f"No install.py is found in {model_path}.", None)
    except subprocess.CalledProcessError as e:
        return (False, e.output, io.FileIO(stdout_fpath, mode="r").read().decode())
    except Exception as e:
        return (False, e, io.FileIO(stdout_fpath, mode="r").read().decode())
    finally:
        del output_buffer
        os.remove(stdout_fpath)

    return (True, None, None)


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
        success, errmsg, stdout_stderr = _install_deps(model_path, verbose=verbose)
        if success:
            print("OK")
        else:
            print("FAIL")
            try:
                errmsg = errmsg.decode()
            except Exception:
                pass

            # If the install was very chatty, we don't want to overwhelm.
            # This will not affect verbose mode, which does not catch stdout
            # and stderr.
            log_lines = (stdout_stderr or "").splitlines(keepends=False)
            if len(log_lines) > 40:
                log_lines = log_lines[:20] + ["..."] + log_lines[-20:]
                stdout_stderr = "\n".join(log_lines)

            if stdout_stderr:
                errmsg = f"{stdout_stderr}\n\n{errmsg or ''}"

            failures[model_path] = errmsg
            if not continue_on_fail:
                break
    for model_path in failures:
        print(f"Error for {model_path}:")
        print("---------------------------------------------------------------------------")
        print(failures[model_path])
        print("---------------------------------------------------------------------------")
        print()

    return len(failures) == 0


@dataclasses.dataclass(frozen=True)
class ModelDetails:
    """Static description of what a particular TorchBench model supports.

    When parameterizing tests, we only want to generate sensible ones.
    (e.g. Those where a model can be imported and supports the feature to be
    tested or benchmarked.) This requires us to import the model; however many
    of the models are EXTREMELY stateful, and even importing them consumes
    significant system resources. As a result, we only want one (or a few)
    alive at any given time.

    This class describes what a particular model does and does not support, so
    that we can release the underlying subprocess but retain any pertinent
    metadata.
    """
    path: str
    exists: bool
    optimized_for_inference: bool
    _diagnostic_msg: str

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


class ModelTask(base_task.TaskBase):

    def __init__(
        self,
        model_path: str,
        timeout: Optional[float] = None,
    ) -> None:
        self._model_path = model_path
        self._worker = subprocess_worker.SubprocessWorker(timeout=timeout)
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

    # =========================================================================
    # == Import Model in the child process ====================================
    # =========================================================================

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

        # Populate global namespace so subsequent calls to worker.run can access `Model`
        globals()["Model"] = Model

        # This will be used to populate a `ModelDetails` instance in the parent.
        return {
            "path": model_path,
            "exists": Model is not None,
            "optimized_for_inference": hasattr(Model, "optimized_for_inference"),
            "_diagnostic_msg": diagnostic_msg,
        }

    # =========================================================================
    # == Instantiate a concrete `model` instance ==============================
    # =========================================================================

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

    def del_model_instance(self):
        self.worker.run("""
            del model
            del maybe_sync
            import gc
            gc.collect()
        """)

    # =========================================================================
    # == Forward calls to `model` from parent to worker =======================
    # =========================================================================

    def set_train(self) -> None:
        self.worker.run("model.set_train()")

    def train(self) -> None:
        self.worker.run("""
            model.train()
            maybe_sync()
        """)

    def set_eval(self) -> None:
        self.worker.run("model.set_eval()")

    def eval(self) -> None:
        self.worker.run("""
            model.eval()
            maybe_sync()
        """)

    def check_opt_vs_noopt_jit(self) -> None:
        self.worker.run("model.check_opt_vs_noopt_jit()")

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def check_example() -> None:
        model = globals()["model"]
        module, example_inputs = model.get_module()
        if isinstance(example_inputs, dict):
            # Huggingface models pass **kwargs as arguments, not *args
            module(**example_inputs)
        else:
            module(*example_inputs)

    # =========================================================================
    # == Control `torch` state (in the subprocess) ============================
    # =========================================================================

    @contextlib.contextmanager
    def no_grad(self, disable_nograd: bool) -> None:
        # TODO: deduplicate with `torchbenchmark.util.model.no_grad`

        initial_value = self.worker.load_stmt("torch.is_grad_enabled()")
        eval_in_nograd = (
            not disable_nograd and
            self.worker.load_stmt("model.eval_in_nograd()"))

        try:
            self.worker.run(f"torch.set_grad_enabled({not eval_in_nograd})")
            yield
        finally:
            self.worker.run(f"torch.set_grad_enabled({initial_value})")

    @contextlib.contextmanager
    def watch_cuda_memory(
        self,
        skip: bool,
        assert_equal: Callable[[int, int], NoReturn],
    ):
        # This context manager is used in testing to ensure we're not leaking
        # memory; these tests are generally parameterized by device, so in some
        # cases we want this (and the outer check) to simply be a no-op.
        if skip:
            yield
            return

        self.worker.run("import gc;gc.collect()")
        memory_before = self.worker.load_stmt("torch.cuda.memory_allocated()")
        yield
        self.worker.run("gc.collect()")
        memory_after = self.worker.load_stmt("torch.cuda.memory_allocated()")
        assert_equal(memory_before, memory_after)
        self.worker.run("torch.cuda.empty_cache()")


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
