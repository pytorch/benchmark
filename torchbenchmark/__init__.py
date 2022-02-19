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
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple
from urllib import request

from components._impl.tasks import base as base_task
from components._impl.workers import subprocess_worker
from .util.env_check import get_pkg_versions

TORCH_DEPS = ['torch', 'torchvision', 'torchtext']
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
        del output_buffer
        os.remove(stdout_fpath)

    return (True, None, None)


def _list_model_paths() -> List[str]:
    p = pathlib.Path(__file__).parent.joinpath(model_dir)
    return sorted(str(child.absolute()) for child in p.iterdir() if child.is_dir())


def setup(models: List[str] = [], verbose: bool = True, continue_on_fail: bool = False) -> bool:
    if not _test_https():
        print(proxy_suggestion)
        sys.exit(-1)

    failures = {}
    models = list(map(lambda p: p.lower(), models))
    model_paths = filter(lambda p: True if not models else os.path.basename(p).lower() in models, _list_model_paths())
    for model_path in model_paths:
        print(f"running setup for {model_path}...", end="", flush=True)
        success, errmsg, stdout_stderr = _install_deps(model_path, verbose=verbose)
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

    Note that affinity cannot be solved by simply calling `torch.set_num_threads`
    in the child process; this will cause PyTorch to use all of the cores but
    at a much lower efficiency.

    This class describes what a particular model does and does not support, so
    that we can release the underlying subprocess but retain any pertinent
    metadata.
    """
    path: str
    exists: bool
    optimized_for_inference: bool
    _diagnostic_msg: str

    metadata: Dict[str, Any]

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


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
        return (
            ["taskset", "--cpu-list", affinity] if affinity else []
        ) + super().args


class ModelTask(base_task.TaskBase):

    # The worker may (and often does) consume significant system resources.
    # In order to ensure that runs do not interfere with each other, we only
    # allow a single ModelTask to exist at a time.
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str,
        timeout: Optional[float] = None,
    ) -> None:
        gc.collect()  # Make sure previous task has a chance to release the lock
        assert self._lock.acquire(blocking=False), "Failed to acquire lock."

        self._model_path = model_path
        self._worker = Worker(timeout=timeout)
        self.worker.run("import torch")

        self._details: ModelDetails = ModelDetails(
            **self._maybe_import_model(
                package=__name__,
                model_path=model_path,
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
            "metadata": {}
        }

    # =========================================================================
    # == Instantiate a concrete `model` instance ==============================
    # =========================================================================

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def make_model_instance(test: str, device: str, jit: bool, batch_size: Optional[int]=None, extra_args: List[str]=[]) -> None:
        Model = globals()["Model"]
        model = Model(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

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

    # =========================================================================
    # == Get Model attribute in the child process =============================
    # =========================================================================
    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def get_model_attribute(attr: str) -> Any:
        model = globals()["model"]
        if hasattr(model, attr):
            return getattr(model, attr)
        else:
            return None

    def gc_collect(self) -> None:
        self.worker.run("""
            import gc
            gc.collect()
        """)

    def del_model_instance(self):
        self.worker.run("""
            del model
            del maybe_sync
        """)
        self.gc_collect()

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

    def extract_details_train(self) -> None:
        self._details.metadata["train_benchmark"] = self.worker.load_stmt("torch.backends.cudnn.benchmark")
        self._details.metadata["train_deterministic"] = self.worker.load_stmt("torch.backends.cudnn.deterministic")

    def check_details_train(self, device, md) -> None:
        self.extract_details_train()
        if device == 'cuda':
            assert md["train_benchmark"] == self._details.metadata["train_benchmark"], \
                "torch.backends.cudnn.benchmark does not match expect metadata during training."
            assert md["train_deterministic"] == self._details.metadata["train_deterministic"], \
                "torch.backends.cudnn.deterministic does not match expect metadata during training."

    def extract_details_eval(self) -> None:
        self._details.metadata["eval_benchmark"] = self.worker.load_stmt("torch.backends.cudnn.benchmark")
        self._details.metadata["eval_deterministic"] = self.worker.load_stmt("torch.backends.cudnn.deterministic")
        # FIXME: Models will use context "with torch.no_grad():", so the lifetime of no_grad will end after the eval().
        # FIXME: Must incorporate this "torch.is_grad_enabled()" inside of actual eval() func.
        # self._details.metadata["eval_nograd"] = not self.worker.load_stmt("torch.is_grad_enabled()")
        self._details.metadata["eval_nograd"] = True

    def check_details_eval(self, device, md) -> None:
        self.extract_details_eval()
        if device == 'cuda':
            assert md["eval_benchmark"] == self._details.metadata["eval_benchmark"], \
                "torch.backends.cudnn.benchmark does not match expect metadata during eval."
            assert md["eval_deterministic"] == self._details.metadata["eval_deterministic"], \
                "torch.backends.cudnn.deterministic does not match expect metadata during eval."
        assert md["eval_nograd"] == self._details.metadata["eval_nograd"], \
            "torch.is_grad_enabled does not match expect metadata during eval."

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

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def check_eval_output() -> None:
        instance = globals()["model"]
        import torch
        out = instance.eval()
        model_name = getattr(instance, 'name', None)
        if not isinstance(out, tuple):
            raise RuntimeError('Model {model_name} eval() output is not a tuple')
        for ind, element in enumerate(out):
            if not isinstance(element, torch.Tensor):
                raise RuntimeError(f'Model {model_name} eval() output is tuple, but'
                                   f' its {ind}-th element is not a Tensor.')

    @base_task.run_in_worker(scoped=True)
    @staticmethod
    def check_device() -> None:
        instance = globals()["model"]

        # Check this BenchmarkModel has a device attribute.
        current_device = getattr(instance, 'device', None)
        if current_device is None:
            raise RuntimeError('Missing device in BenchmarkModel.')

        model, inputs = instance.get_module()
        model_name = getattr(model, 'name', None)

        # Check the model tensors are assigned to the expected device.
        for t in model.parameters():
            model_device = t.device.type
            if model_device != current_device:
                raise RuntimeError(f'Model {model_name} was not set to the'
                                   f' expected device {current_device},'
                                   f' found device {model_device}.')

        # Check the inputs are assigned to the expected device.
        def check_inputs(inputs):
            if isinstance(inputs, torch.Tensor):
                if inputs.dim() and current_device == "cuda":
                    # Zero dim Tensors (Scalars) can be captured by CUDA
                    # kernels and need not match device.
                    return

                inputs_device = inputs.device.type
                if inputs_device != current_device:
                    raise RuntimeError(f'Model {model_name} inputs were'
                                       f' not set to the expected device'
                                       f' {current_device}, found device'
                                       f' {inputs_device}.')
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

        self.gc_collect()
        memory_before = self.worker.load_stmt("torch.cuda.memory_allocated()")
        yield
        self.gc_collect()
        assert_equal(
            memory_before,
            self.worker.load_stmt("torch.cuda.memory_allocated()"),
        )
        self.worker.run("torch.cuda.empty_cache()")


def list_models_details(workers: int = 1) -> List[ModelDetails]:
    return [
        ModelTask(model_path).model_details
        for model_path in _list_model_paths()
    ]


def list_models(model_match=None):
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

        # If given model_match, only return full or partial name matches in models.
        if model_match is None:
            models.append(Model)
        else:
            if model_match.lower() in Model.name.lower():
                models.append(Model)
    return models

def load_model_by_name(model):
    models = filter(lambda x: model.lower() == x.lower(),
                    map(lambda y: os.path.basename(y), _list_model_paths()))
    models = list(models)
    if not models:
        return None
    assert len(models) == 1, f"Found more than one models {models} with the exact name: {model}"
    model_name = models[0]
    try:
        module = importlib.import_module(f'.models.{model_name}', package=__name__)
    except ModuleNotFoundError as e:
        print(f"Warning: Could not find dependent module {e.name} for Model {model_name}, skip it")
        return None
    Model = getattr(module, 'Model', None)
    if Model is None:
        print(f"Warning: {module} does not define attribute Model, skip it")
        return None
    if not hasattr(Model, 'name'):
        Model.name = model_name
    return Model

def get_metadata_from_yaml(path):
    import yaml
    metadata_path = path + "/metadata.yaml"
    md = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            md = yaml.load(f, Loader=yaml.FullLoader)
    return md

def str_to_bool(input: Any) -> bool:
    if not input:
        return False
    return str(input).lower() in ("1", "yes", "y", "true", "t", "on")
