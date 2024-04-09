import importlib
import torch
from contextlib import contextmanager, ExitStack
import warnings
import yaml
import time
from pathlib import Path
from typing import ContextManager, Optional, List, Tuple, Generator
from torchbenchmark import REPO_PATH
from torchbenchmark.util.extra_args import parse_opt_args, apply_opt_args, \
                                           parse_decoration_args, apply_decoration_args, is_staged_train_test, \
                                           TEST_STAGE
from torchbenchmark.util.env_check import set_random_seed, is_hf_model, \
                                          save_deterministic_dict, load_deterministic_dict, check_accuracy
from torchbenchmark.util.fx_int8 import get_sub_module, prepare_sub_module, convert_sub_module
from torchbenchmark.util.input import input_cast, ModelInputDescriptor

SPECIAL_DEVICE_MAPPING = {
    "AMD Instinct MI210": "NVIDIA A100-SXM4-40GB"
}

class PostInitProcessor(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post__init__()
        return obj

@contextmanager
def no_grad(val):
    """Some meta-learning models (e.g. maml) may need to train a target(another) model
    in inference runs
    """
    old_state = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(not val)
        yield
    finally:
        torch.set_grad_enabled(old_state)

@contextmanager
def nested(*contexts):
    """
    Chain and apply a list of contexts
    """
    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield contexts

# enable JIT profiling executor
@contextmanager
def enable_profiling_executor():
    try:
        graph_executor = torch._C._get_graph_executor_optimize(True)
        profiling_executor = torch._C._jit_set_profiling_executor(True)
        profiling_mode = torch._C._jit_set_profiling_mode(True)
        yield
    finally:
        torch._C._jit_set_profiling_mode(profiling_mode)
        torch._C._jit_set_profiling_executor(profiling_executor)
        torch._C._get_graph_executor_optimize(graph_executor)

class BenchmarkModel(metaclass=PostInitProcessor):
    DEFAULT_TRAIN_BSIZE: Optional[int] = None
    DEFAULT_EVAL_BSIZE: Optional[int] = None
    # by default, deepcopy the model when checking accuracy
    # because some models are stateful (such as moco)
    DEEPCOPY: bool = True
    # by default, turn on deterministic mode when checking accuracy
    DISABLE_DETERMINISM: bool = False

    test: str
    device: str
    batch_size: int
    extra_args: List[str]
    run_contexts: List[ContextManager]

    """
    A base class for adding models to torch benchmark.
    See [Adding Models](#../models/ADDING_MODELS.md)
    """
    def __init__(self, test: str, device: str, batch_size: Optional[int]=None, extra_args: List[str]=[]):
        self._start_init_time = time.time_ns()
        self.metadata = self._load_metadata()
        self.test = test
        # sanity checks of the options
        assert self.test == "train" or self.test == "eval", \
               f"Test must be 'train' or 'eval', but provided {self.test}."
        self.device = device
        self.extra_args = extra_args
        self.opt = None
        self._skip_by_device_name()
        # contexts to run in the test function
        if self.test == "train":
            # In train test, there are run contexts that should only be applied for forward/backward/optimizer stage
            # For example, amp only applies for the forward stage
            self.forward_contexts = []
            self.backward_contexts = []
            self.optimizer_contexts = []
        self.run_contexts = [
            enable_profiling_executor  # force JIT profiling executor to be enabled by default
        ]

        set_random_seed()
        # parse the args
        self.dargs, opt_args = parse_decoration_args(self, self.extra_args)
        if self.dargs.accuracy and not self.DISABLE_DETERMINISM:
            self.deterministic_dict = save_deterministic_dict(self.name)
        # if the args contain "--torchdynamo", parse torchdynamo args
        if "--torchdynamo" in opt_args or "--inductor" in opt_args:
            self.dynamo = True
            from torchbenchmark.util.backends.torchdynamo import parse_torchdynamo_args
            self.opt_args, self.extra_args = parse_torchdynamo_args(opt_args)
        else:
            self.dynamo = False
            self.opt_args, self.extra_args = parse_opt_args(self, opt_args)
        self._determine_batch_size(batch_size)
        self.num_batch = self._determine_dynamic_num_batches(self.dargs.num_batch)

    # Run the post processing for model acceleration
    def __post__init__(self):
        # All arguments should be parsed at this point.
        assert not self.extra_args, f"Expected no unknown args at this point, found {self.extra_args}"
        if self.dargs.accuracy:
            self.accuracy = check_accuracy(self)
            if not self.DISABLE_DETERMINISM:
                load_deterministic_dict(self.deterministic_dict)
            return
        # apply decoration args
        apply_decoration_args(self, self.dargs)
        # apply optimization args
        if self.dynamo:
            from torchbenchmark.util.backends.torchdynamo import apply_torchdynamo_args
            apply_torchdynamo_args(self, self.opt_args, self.dargs.precision)
        else:
            apply_opt_args(self, self.opt_args)
        # setup distributed trainer
        if self.dargs.distributed:
            if self.dargs.distributed_wrap_fn:
                pos = self.dargs.distributed_wrap_fn.rfind(".")
                module = importlib.import_module(self.dargs.distributed_wrap_fn[:pos])
                apply_trainer = getattr(module, self.dargs.distributed_wrap_fn[(pos+1):])
            else:
                from torchbenchmark.util.distributed.core_model.apply_trainer import apply_trainer
            if is_hf_model(self):
                # DDP requires to use unwrapped model for huggingface
                module, _inputs = self.get_module(wrap_model=False)
            else:
                module, _inputs = self.get_module()
            self.set_module(apply_trainer(module, self.dargs.distributed))
        # Need to clean up the cache because we run deep copy within correceness check
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self._end_init_time = time.time_ns()

    def _skip_by_device_name(self):
        if not self.device == "cuda":
            return
        current_device_name = torch.cuda.get_device_name()
        if self.metadata and "not_implemented" in self.metadata and self.metadata["not_implemented"]:
            for skip in self.metadata["not_implemented"]:
                skip_test = skip["test"] if "test" in skip else None
                skip_device = skip["device"] if "device" in skip else None
                if skip_device == current_device_name and (not skip_test or skip_test == self.test):
                    raise NotImplementedError(f"The current device {current_device_name} is skipped by its `{self.name}/metadata.yaml`.")

    def _determine_dynamic_num_batches(self, user_specified_num_batches: Optional[int]) -> int:
        if user_specified_num_batches and not user_specified_num_batches == 1:
            assert self.test == "train", "Only train test support multiple batches at this moment."
            return user_specified_num_batches
        # If user does not specify num_batch, run a single batch by default
        return 1

    def _get_batch_size_from_metadata(self) -> Optional[str]:
        if self.device != "cuda":
            current_device_name = str(self.device)
        else:
            current_device_name = torch.cuda.get_device_name() if torch.cuda.get_device_name() else "UNKNOWN"
            if current_device_name in SPECIAL_DEVICE_MAPPING:
                current_device_name = SPECIAL_DEVICE_MAPPING[current_device_name]

        # use the device suggestion on CUDA inference tests, key should be either eval_batch_size or train_batch_size
        device_batch_size_key = f"{self.test}_batch_size"
        if self.metadata and "devices" in self.metadata and current_device_name in self.metadata["devices"] \
                            and device_batch_size_key in self.metadata["devices"][current_device_name]:
            batch_size = self.metadata["devices"][current_device_name][device_batch_size_key]
            return batch_size

    def _determine_batch_size(self, user_specified_batch_size=None):
        self.batch_size = user_specified_batch_size

        if not self.batch_size and getattr(self, "ALLOW_CUSTOMIZE_BSIZE", True):
            device_specified_batch_size = self._get_batch_size_from_metadata()
            self.batch_size = device_specified_batch_size

        if not self.batch_size:
            default_batch_size = self.DEFAULT_TRAIN_BSIZE if self.test == "train" else self.DEFAULT_EVAL_BSIZE
            self.batch_size = default_batch_size

        if not self.batch_size:
            raise NotImplementedError(f"Model's {'DEFAULT_TRAIN_BSIZE' if self.test == 'train' else 'DEFAULT_EVAL_BSIZE'} is not implemented.")

        # Check if specified batch size is supported by the model
        if not getattr(self, "ALLOW_CUSTOMIZE_BSIZE", True):
            if self.test == "train" and (not self.batch_size == self.DEFAULT_TRAIN_BSIZE):
                raise NotImplementedError(f"Model doesn't support customizing batch size, but {self.test} test is providing a batch size other than DEFAULT_TRAIN_BSIZE")
            elif self.test == "eval" and (not self.batch_size == self.DEFAULT_EVAL_BSIZE):
                raise NotImplementedError(f"Model doesn't support customizing batch size, but {self.test} test is providing a batch size other than DEFAULT_EVAL_BSIZE")
        elif self.dargs.accuracy:
            self.batch_size = 4 if self.batch_size > 4 else self.batch_size

    def _load_metadata(self):
        relative_path = self.__class__.__module__.split(".")
        if getattr(self, "name", None) == None:
            self.name = relative_path[-1]
        metadata_loc = Path(REPO_PATH).joinpath(*relative_path).joinpath("metadata.yaml")
        if not metadata_loc.exists():
            return None
        with open(metadata_loc, "r") as mf:
            metadata = yaml.safe_load(mf)
        return metadata

    def add_context(self, context_fn, stage=TEST_STAGE.ALL):
        ctx = context_fn()
        assert isinstance(ctx, ContextManager), f"Expected adding a ContextManager, get {type(ctx)}. Please report a bug."
        if stage == TEST_STAGE.ALL:
            self.run_contexts.append(context_fn)
        elif stage == TEST_STAGE.FORWARD:
            self.forward_contexts.append(context_fn)
        elif stage == TEST_STAGE.BACKWARD:
            self.backward_contexts.append(context_fn)
        elif stage == TEST_STAGE.OPTIMIZER:
            self.optimizer_contexts.append(context_fn)

    # Common interface for all models extending BenchmarkModel to access the optimizer.
    # Some models have an opt attribute, others have an optimizer attribute; this
    # implementation handles both. This function should not error! Simply return None
    # if there's no optimizer in sight.
    def get_optimizer(self):
        if hasattr(self, "optimizer"):
            return self.optimizer
        if hasattr(self, "opt"):
            return self.opt
        warnings.warn("The optimizer for this model is not stored in self.opt nor self.optimizer. "
                      "Currently returning None! Please override this implementation with your own "
                      "if there is an optimizer this should be returning instead.")
        return None

    # Takes in an optimizer and sets that to be the optimizer used from now on.
    # There are special models like dcgan that would update multiple optimizers at once,
    # so optimizer here is not always strictly a, say, torch.optim.Optimizer.
    def set_optimizer(self, optimizer) -> None:
        if hasattr(self, "optimizer"):
            self.optimizer = optimizer
            return
        if hasattr(self, "opt"):
            self.opt = optimizer
            return
        raise NotImplementedError("The optimizer for this model is not stored in self.opt nor self.optimizer. "
                                  "Please override this implementation with your own.")

    # Default implementation for replacing the model
    def set_module(self, new_model):
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            self.model = new_model
        else:
            raise NotImplementedError("The instance variable 'model' does not exist or is not type 'torch.nn.Module', implement your own `set_module()` function.")

    def get_input_iter(self) -> Generator:
        """Return the dynamic input iterator for the model. By default, always return the same batch of input."""
        model, example_inputs = self.get_module()
        while True:
            yield example_inputs

    def get_input_descriptor(self) -> ModelInputDescriptor:
        if hasattr(self, 'input_descriptor') and isinstance(self.input_descriptor, ModelInputDescriptor):
            return self.input_descriptor
        raise NotImplementedError(f"Default dynamic input descriptor is not implemented. "
                                  "Please submit an issue if you need a dynamic shape input iterator implementation for the model {self.name}.")

    def set_input_descriptor(self, descriptor: ModelInputDescriptor) -> None:
        """Set the customized dynamic input descriptor for the model."""
        if hasattr(self, 'input_descriptor') and isinstance(self.input_descriptor, ModelInputDescriptor):
            self.input_descriptor = descriptor
            return
        raise NotImplementedError(f"Default dynamic input descriptor is not implemented."
                                  "Please submit an issue if you need a dynamic shape input descriptor implementation for the model {self.name}.")

    def _invoke_staged_train_test(self, num_batch: int) -> None:
        optimizer = self.get_optimizer()
        input_generator = self.get_input_iter() if not num_batch == 1 else None
        for _batch_num in range(num_batch):
            if input_generator:
                self.example_inputs = next(input_generator)
                # cast inputs if needed
                apply_decoration_args(self, self.dargs)
            if optimizer is not None:
                optimizer.zero_grad()
            with nested(*self.forward_contexts):
                losses = self.forward()
            with nested(*self.backward_contexts):
                self.backward(losses)
            if optimizer is not None:
                with nested(*self.optimizer_contexts):
                    self.optimizer_step()
        return None

    def invoke(self) -> Optional[Tuple[torch.Tensor]]:
        if self.test == "train" and is_staged_train_test(self) and (getattr(self, "train", None) == None):
            return self._invoke_staged_train_test(num_batch=self.num_batch)
        assert self.num_batch == 1, "Only staged_train_test supports multiple-batch testing at this time."
        out = None
        with nested(*self.run_contexts):
            if self.test == "train":
                self.train()
            elif self.test == "eval":
                out = self.eval()
        return out

    def eval_in_nograd(self):
        return True

    def enable_fx_int8(self, quant_engine:str='x86'):
        torch.backends.quantized.engine = quant_engine
        try:
            model, _ = self.get_module()
            # Get sub modules
            model, sub_module_list = get_sub_module(model, dict(model.named_modules()), '')
            if not len(sub_module_list):
                warnings.warn(UserWarning(f"{self.name} doesn't have submodule can ben quantized!"))
            model = prepare_sub_module(sub_module_list, model, '', quant_engine)
            self.set_module(model)
            # Calibration
            self.eval()
            model, _ = self.get_module()
            model = convert_sub_module(sub_module_list, model, '')
            self.set_module(model)
        except Exception as e:
            print(e)
            raise RuntimeError(f"{self.name} doesn't support `fx_int8` yet!")

    def _cast_to(self, cond, action):
        model_name = self.name
        try:
            model, _ = self.get_module()
            model = action(model)
        except RuntimeError:
            warnings.warn(UserWarning(f"{model_name} doesn't support cast to {action} yet!"))
            return
        self.set_module(model)
        if hasattr(self, 'example_inputs'):
            self.example_inputs = input_cast(cond, action, self.example_inputs)
        else:
            warnings.warn(UserWarning(f"{model_name} example inputs doesn't cast to {action} yet!"))

    def enable_bf16(self):
        tensor_cond = lambda x: x.dtype == torch.float32
        tensor_action = lambda x: x.to(torch.bfloat16)
        self._cast_to(tensor_cond, tensor_action)

    def enable_fp16(self):
        tensor_cond = lambda x: x.dtype == torch.float32
        tensor_action = lambda x: x.half()
        self._cast_to(tensor_cond, tensor_action)

    def enable_channels_last(self):
        tensor_cond = lambda x: x.dim() == 4
        tensor_action = lambda x: x.to(memory_format=torch.channels_last)
        self._cast_to(tensor_cond, tensor_action)

    def enable_amp(self):
        if not self.dynamo and self.opt_args.backend == 'cudagraph':
            return NotImplementedError("AMP not implemented for cudagraphs")
        if not hasattr(self, "amp_context"):
            raise RuntimeError(f"{self.name} doesn't have amp_context support!")
        if self.device == "cpu":
            self.amp_context = lambda: torch.cpu.amp.autocast()
        elif self.device == "cuda":
            self.amp_context = lambda: torch.cuda.amp.autocast()
        if is_staged_train_test(self):
            self.forward_contexts.append(self.amp_context)

    @property
    def pt2_compilation_time(self) -> Optional[float]:
        from torch._dynamo.utils import compile_times
        compile_time = dict(zip(*compile_times(repr="csv", aggregate=True)))
        if "_compile.<locals>.compile_inner" in compile_time:
            return float(compile_time["_compile.<locals>.compile_inner"])
        return None

    @property
    def pt2_graph_breaks(self) -> int:
        from torch._dynamo.utils import counters
        num_graph_breaks = len(counters["graph_break"].keys())
        return num_graph_breaks

    @property
    def ttfb(self) -> float:
        """Return the time taken to the first batch in ms."""
        return (self._end_init_time - self._start_init_time) / 1_000_000
