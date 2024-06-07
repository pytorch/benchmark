import contextlib
import dataclasses
import gc
import importlib
import io
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple

import torch

from . import canary_models, e2e_models, models, util

from ._components._impl.tasks import base as base_task
from ._components._impl.workers import subprocess_worker


class ModelNotFoundError(RuntimeError):
    pass


REPO_PATH = Path(os.path.abspath(__file__)).parent.parent
DATA_PATH = os.path.join(REPO_PATH, "torchbenchmark", "data", ".data")


class add_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


with add_path(str(REPO_PATH)):
    from utils import get_pkg_versions, TORCH_DEPS

this_dir = pathlib.Path(__file__).parent.absolute()
model_dir = "models"
internal_model_dir = "fb"
canary_model_dir = "canary_models"
install_file = "install.py"


def _install_deps(model_path: str, verbose: bool = True) -> Tuple[bool, Any]:
    from .util.env_check import get_pkg_versions

    run_args = [
        [sys.executable, install_file],
    ]
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = Path(this_dir.parent).as_posix()
    run_kwargs = {
        "cwd": model_path,
        "check": True,
        "env": run_env,
    }

    output_buffer = None
    fd, stdout_fpath = tempfile.mkstemp()

    try:
        output_buffer = io.FileIO(stdout_fpath, mode="w")
        if os.path.exists(os.path.join(model_path, install_file)):
            if not verbose:
                run_kwargs["stderr"] = subprocess.STDOUT
                run_kwargs["stdout"] = output_buffer
            versions = get_pkg_versions(TORCH_DEPS)
            subprocess.run(*run_args, **run_kwargs)  # type: ignore
            new_versions = get_pkg_versions(TORCH_DEPS)
            if versions != new_versions:
                errmsg = f"The torch packages are re-installed after installing the benchmark deps. \
                           Before: {versions}, after: {new_versions}"
                return (False, errmsg, None)
        else:
            return (True, f"No install.py is found in {model_path}. Skip.", None)
    except subprocess.CalledProcessError as e:
        return (False, e.output, io.FileIO(stdout_fpath, mode="r").read().decode())
    except Exception as e:
        return (False, e, io.FileIO(stdout_fpath, mode="r").read().decode())
    finally:
        output_buffer.close()
        del output_buffer
        os.close(fd)
        os.remove(stdout_fpath)

    return (True, None, None)


def dir_contains_file(dir, file_name) -> bool:
    names = map(lambda x: x.name, filter(lambda x: x.is_file(), dir.iterdir()))
    return file_name in names


def _list_model_paths(internal=True) -> List[str]:
    p = pathlib.Path(__file__).parent.joinpath(model_dir)
    # Only load the model directories that contain a "__init.py__" file
    models = sorted(
        str(child.absolute())
        for child in p.iterdir()
        if child.is_dir()
        and (not child.name == internal_model_dir)
        and dir_contains_file(child, "__init__.py")
    )
    p = p.joinpath(internal_model_dir)
    if p.exists() and internal:
        m = sorted(
            str(child.absolute())
            for child in p.iterdir()
            if child.is_dir() and dir_contains_file(child, "__init__.py")
        )
        models.extend(m)
    return models


def _list_canary_model_paths() -> List[str]:
    p = pathlib.Path(__file__).parent.joinpath(canary_model_dir)
    # Only load the model directories that contain a "__init.py__" file
    models = sorted(
        str(child.absolute())
        for child in p.iterdir()
        if child.is_dir()
        and (not child.name == internal_model_dir)
        and dir_contains_file(child, "__init__.py")
    )
    return models


def _is_internal_model(model_name: str) -> bool:
    p = (
        pathlib.Path(__file__)
        .parent.joinpath(model_dir)
        .joinpath(internal_model_dir)
        .joinpath(model_name)
    )
    if p.exists() and p.joinpath("__init__.py").exists():
        return True
    return False


def _is_canary_model(model_name: str) -> bool:
    p = pathlib.Path(__file__).parent.joinpath(canary_model_dir).joinpath(model_name)
    if p.exists() and p.joinpath("__init__.py").exists():
        return True
    return False


def setup(
    models: List[str] = [],
    verbose: bool = True,
    continue_on_fail: bool = False,
    test_mode: bool = False,
    allow_canary: bool = False,
) -> bool:
    failures = {}
    models = list(map(lambda p: p.lower(), models))
    model_paths = filter(
        lambda p: True if not models else os.path.basename(p).lower() in models,
        _list_model_paths(),
    )
    if allow_canary:
        canary_model_paths = filter(
            lambda p: True if not models else os.path.basename(p).lower() in models,
            _list_canary_model_paths(),
        )
        model_paths = list(model_paths)
        model_paths.extend(canary_model_paths)
    for model_path in model_paths:
        print(f"running setup for {model_path}...", end="", flush=True)
        if test_mode:
            versions = get_pkg_versions(TORCH_DEPS)
        success, errmsg, stdout_stderr = _install_deps(model_path, verbose=verbose)
        if test_mode:
            new_versions = get_pkg_versions(TORCH_DEPS, reload=True)
            if versions != new_versions:
                print(
                    f"The torch packages are re-installed after installing the benchmark model {model_path}. \
                        Before: {versions}, after: {new_versions}"
                )
                sys.exit(-1)
        if success and errmsg and "No install.py is found" in errmsg:
            print("SKIP - No install.py is found")
        elif success:
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
        print(
            "---------------------------------------------------------------------------"
        )
        print(failures[model_path])
        print(
            "---------------------------------------------------------------------------"
        )
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

    Note that affinity cannot be solved by simply calling `torch.set_num_threads`
    in the child process; this will cause PyTorch to use all of the cores but
    at a much lower efficiency.

    This class describes what a particular model does and does not support, so
    that we can release the underlying subprocess but retain any pertinent
    metadata.
    """

    name: str
    exists: bool
    _diagnostic_msg: str

    metadata: Dict[str, Any]


class Worker(subprocess_worker.SubprocessWorker):
    """Run subprocess using taskset if CPU affinity is set.

    When GOMP_CPU_AFFINITY is set, importing `torch` in the main process has
    the very surprising effect of changing the threading behavior in the
    subprocess. (See https://github.com/pytorch/pytorch/issues/49971 for
    details.) This is a problem, because it means that the worker is not
    hermetic and also tends to force the subprocess torch to run in single
    threaded mode which drastically skews results.

    This can be ameliorated by calling the subprocess using `taskset`, which
    allows the subprocess PyTorch to properly bind threads.
    """

    @property
    def args(self) -> List[str]:
        affinity = os.environ.get("GOMP_CPU_AFFINITY", "")
        return (["taskset", "--cpu-list", affinity] if affinity else []) + super().args


class ModelTask(base_task.TaskBase):

    # The worker may (and often does) consume significant system resources.
    # In order to ensure that runs do not interfere with each other, we only
    # allow a single ModelTask to exist at a time.
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str,
        timeout: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
        save_output_dir: Optional[pathlib.Path] = None,
    ) -> None:
        gc.collect()  # Make sure previous task has a chance to release the lock
        assert self._lock.acquire(blocking=False), "Failed to acquire lock."

        self._model_name = model_name
        self._worker = Worker(
            timeout=timeout, extra_env=extra_env, save_output_dir=save_output_dir
        )

        self.worker.run("import torch")
        self._details: ModelDetails = ModelDetails(
            **self._maybe_import_model(
                package=__name__,
                model_name=model_name,
            )
        )

    def __del__(self) -> None:
        self._lock.release()

    @property
    def worker(self) -> subprocess_worker.SubprocessWorker:
        return self._worker

    @property
    def model_details(self) -> bool:
        return self._details

    def __str__(self) -> str:
        return f"ModelTask(Model Name: {self._model_name}, Metadata: {self._details.metadata})"

    # =========================================================================
    # == Import Model in the child process ====================================
    # =========================================================================

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def _maybe_import_model(package: str, model_name: str) -> Dict[str, Any]:
        import importlib
        import os
        import traceback
        from torchbenchmark import load_model_by_name

        diagnostic_msg = ""
        Model = load_model_by_name(model_name)

        # Populate global namespace so subsequent calls to worker.run can access `Model`
        globals()["Model"] = Model

        # This will be used to populate a `ModelDetails` instance in the parent.
        return {
            "name": model_name,
            "exists": Model is not None,
            "_diagnostic_msg": diagnostic_msg,
            "metadata": {},
        }

    # =========================================================================
    # == Instantiate a concrete `model` instance ==============================
    # =========================================================================

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def make_model_instance(
        test: str,
        device: str,
        batch_size: Optional[int] = None,
        extra_args: List[str] = [],
    ) -> None:
        Model = globals()["Model"]
        model = Model(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )

        import gc

        gc.collect()

        if device == "cuda":
            torch.cuda.empty_cache()
            maybe_sync = torch.cuda.synchronize
        else:
            maybe_sync = lambda: None

        globals().update(
            {
                "model": model,
                "maybe_sync": maybe_sync,
            }
        )

    # =========================================================================
    # == Get Model attribute in the child process =============================
    # =========================================================================
    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def get_model_attribute(
        attr: str, field: str = None, classattr: bool = False
    ) -> Any:
        if classattr:
            model = globals()["Model"]
        else:
            model = globals()["model"]
        if hasattr(model, attr):
            if field:
                model_attr = getattr(model, attr)
                return getattr(model_attr, field)
            else:
                return getattr(model, attr)
        else:
            return None

    def gc_collect(self) -> None:
        self.worker.run(
            """
            import gc
            gc.collect()
        """
        )

    def del_model_instance(self):
        self.worker.run(
            """
            del model
            del maybe_sync
        """
        )
        self.gc_collect()

    # =========================================================================
    # == Forward calls to `model` from parent to worker =======================
    # =========================================================================

    def set_train(self) -> None:
        self.worker.run("model.set_train()")

    def invoke(self) -> None:
        self.worker.run(
            """
            model.invoke()
            maybe_sync()
        """
        )

    def set_eval(self) -> None:
        self.worker.run("model.set_eval()")

    def extract_details_train(self) -> None:
        self._details.metadata["train_benchmark"] = self.worker.load_stmt(
            "torch.backends.cudnn.benchmark"
        )
        self._details.metadata["train_deterministic"] = self.worker.load_stmt(
            "torch.backends.cudnn.deterministic"
        )

    def check_details_train(self, device, md) -> None:
        self.extract_details_train()
        if device == "cuda":
            assert (
                md["train_benchmark"] == self._details.metadata["train_benchmark"]
            ), "torch.backends.cudnn.benchmark does not match expect metadata during training."
            assert (
                md["train_deterministic"]
                == self._details.metadata["train_deterministic"]
            ), "torch.backends.cudnn.deterministic does not match expect metadata during training."

    def extract_details_eval(self) -> None:
        self._details.metadata["eval_benchmark"] = self.worker.load_stmt(
            "torch.backends.cudnn.benchmark"
        )
        self._details.metadata["eval_deterministic"] = self.worker.load_stmt(
            "torch.backends.cudnn.deterministic"
        )
        # FIXME: Models will use context "with torch.no_grad():", so the lifetime of no_grad will end after the eval().
        # FIXME: Must incorporate this "torch.is_grad_enabled()" inside of actual eval() func.
        # self._details.metadata["eval_nograd"] = not self.worker.load_stmt("torch.is_grad_enabled()")
        self._details.metadata["eval_nograd"] = True

    def check_details_eval(self, device, md) -> None:
        self.extract_details_eval()
        if device == "cuda":
            assert (
                md["eval_benchmark"] == self._details.metadata["eval_benchmark"]
            ), "torch.backends.cudnn.benchmark does not match expect metadata during eval."
            assert (
                md["eval_deterministic"] == self._details.metadata["eval_deterministic"]
            ), "torch.backends.cudnn.deterministic does not match expect metadata during eval."
        assert (
            md["eval_nograd"] == self._details.metadata["eval_nograd"]
        ), "torch.is_grad_enabled does not match expect metadata during eval."

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def check_eval_output() -> None:
        instance = globals()["model"]
        assert (
            instance.test == "eval"
        ), "We only support checking output of an eval test. Please submit a bug report."
        instance.invoke()

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def check_device() -> None:
        instance = globals()["model"]

        # Check this BenchmarkModel has a device attribute.
        current_device = getattr(instance, "device", None)
        if current_device is None:
            raise RuntimeError("Missing device in BenchmarkModel.")

        model, inputs = instance.get_module()
        # test set_module
        instance.set_module(model)
        model_name = instance.name

        # Check the model tensors are assigned to the expected device.
        for t in model.parameters():
            model_device = t.device.type
            if model_device != current_device:
                raise RuntimeError(
                    f"Model {model_name} was not set to the"
                    f" expected device {current_device},"
                    f" found device {model_device}."
                )

        # Check the inputs are assigned to the expected device.
        def check_inputs(inputs):
            if isinstance(inputs, torch.Tensor):
                if inputs.dim() and current_device == "cuda":
                    # Zero dim Tensors (Scalars) can be captured by CUDA
                    # kernels and need not match device.
                    return

                inputs_device = inputs.device.type
                if inputs_device != current_device:
                    raise RuntimeError(
                        f"Model {model_name} inputs were"
                        f" not set to the expected device"
                        f" {current_device}, found device"
                        f" {inputs_device}."
                    )
            elif isinstance(inputs, tuple):
                # Some inputs are nested inside tuples, such as tacotron2
                for i in inputs:
                    check_inputs(i)
            elif isinstance(inputs, dict):
                # Huggingface models take inputs as kwargs
                for i in inputs.values():
                    check_inputs(i)

        check_inputs(inputs)

    # =========================================================================
    # == Control `torch` state (in the subprocess) ============================
    # =========================================================================

    @contextlib.contextmanager
    def watch_cuda_memory(
        self,
        skip: bool,
        assert_equal: Callable[[int, int], NoReturn],
    ):
        # This context manager is used in testing to ensure we're not leaking
        # memory; these tests are generally parameterized by device, so in some
        # cases we want this (and the outer check) to simply be a no-op.
        if skip or os.getenv("PYTORCH_TEST_SKIP_CUDA_MEM_LEAK_CHECK", "0") == "1":
            yield
            return
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            self.worker.load_stmt("torch._C._cuda_clearCublasWorkspaces()")
        self.gc_collect()
        memory_before = self.worker.load_stmt("torch.cuda.memory_allocated()")
        yield
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            self.worker.load_stmt("torch._C._cuda_clearCublasWorkspaces()")
        self.gc_collect()
        assert_equal(
            memory_before,
            self.worker.load_stmt("torch.cuda.memory_allocated()"),
        )
        self.worker.run("torch.cuda.empty_cache()")


def list_models_details(workers: int = 1) -> List[ModelDetails]:
    return [ModelTask(os.path.basename(model_path)).model_details for model_path in _list_model_paths()]


def list_models(model_match=None):
    models = []
    for model_path in _list_model_paths():
        model_name = os.path.basename(model_path)
        model_pkg = (
            model_name
            if not _is_internal_model(model_name)
            else f"{internal_model_dir}.{model_name}"
        )
        try:
            module = importlib.import_module(f".models.{model_pkg}", package=__name__)
        except ModuleNotFoundError as e:
            print(
                f"Warning: Could not find dependent module {e.name} for Model {model_name}, skip it"
            )
            continue
        Model = getattr(module, "Model", None)
        if Model is None:
            print(f"Warning: {module} does not define attribute Model, skip it")
            continue
        if not hasattr(Model, "name"):
            Model.name = model_name

        # If given model_match, only return full or partial name matches in models.
        if model_match is None:
            models.append(Model)
        else:
            if model_match.lower() in Model.name.lower():
                models.append(Model)
    return models


def load_model_by_name(model_name: str):
    models = filter(
        lambda x: model_name.lower() == x.lower(),
        map(lambda y: os.path.basename(y), _list_model_paths()),
    )
    models = list(models)
    cls_name = "Model"
    if not models:
        # If the model is in TIMM or Huggingface extended model list
        from torchbenchmark.util.framework.huggingface.list_extended_configs import (
            list_extended_huggingface_models,
        )
        from torchbenchmark.util.framework.timm.extended_configs import (
            list_extended_timm_models,
        )

        if model_name in list_extended_huggingface_models():
            cls_name = "ExtendedHuggingFaceModel"
            module_path = ".util.framework.huggingface.model_factory"
            models.append(model_name)
        elif model_name in list_extended_timm_models():
            cls_name = "ExtendedTimmModel"
            module_path = ".util.framework.timm.model_factory"
            models.append(model_name)
        else:
            raise ModelNotFoundError(
                f"{model_name} is not found in the core model list."
            )
    else:
        model_name = models[0]
        model_pkg = (
            model_name
            if not _is_internal_model(model_name)
            else f"{internal_model_dir}.{model_name}"
        )
        module_path = f".models.{model_pkg}"
    assert (
        len(models) == 1
    ), f"Found more than one models {models} with the exact name: {model_name}"

    module = importlib.import_module(module_path, package=__name__)
    if accelerator_backend := os.getenv("ACCELERATOR_BACKEND"):
        setattr(
            module,
            accelerator_backend,
            importlib.import_module(accelerator_backend),
        )
    Model = getattr(module, cls_name, None)
    if Model is None:
        print(f"Warning: {module} does not define attribute Model, skip it")
        return None
    if not hasattr(Model, "name"):
        Model.name = model_name
    return Model


def load_canary_model_by_name(model: str):
    if not _is_canary_model(model):
        raise ModelNotFoundError(f"{model} is not found in the canary model list.")
    module = importlib.import_module(f".canary_models.{model}", package=__name__)
    Model = getattr(module, "Model", None)
    if Model is None:
        print(f"Warning: {module} does not define attribute Model, skip it")
        return None
    if not hasattr(Model, "name"):
        Model.name = model
    return Model


def get_metadata_from_yaml(path):
    import yaml

    metadata_path = path + "/metadata.yaml"
    md = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            md = yaml.load(f, Loader=yaml.FullLoader)
    return md


def str_to_bool(input: Any) -> bool:
    if not input:
        return False
    return str(input).lower() in ("1", "yes", "y", "true", "t", "on")
