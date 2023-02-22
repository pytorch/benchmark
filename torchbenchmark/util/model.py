import copy
import importlib
import os
import torch
from contextlib import contextmanager, ExitStack
import warnings
import inspect
import yaml
from pathlib import Path
from typing import ContextManager, Optional, List, Tuple, Generator
from torch.utils._pytree import tree_map
from torchbenchmark import REPO_PATH
from torchbenchmark.util.extra_args import check_correctness_p, parse_opt_args, apply_opt_args, \
                                           parse_decoration_args, apply_decoration_args, is_staged_train_test, \
                                           TEST_STAGE
from torchbenchmark.util.env_check import set_random_seed, correctness_check, stableness_check, is_hf_model

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
    # by default, deepcopy the model when checking correctness
    # because some models are stateful (such as moco)
    DEEPCOPY: bool = True

    test: str
    device: str
    jit: bool
    batch_size: int
    extra_args: List[str]
    run_contexts: List[ContextManager]

    """
    A base class for adding models to torch benchmark.
    See [Adding Models](#../models/ADDING_MODELS.md)
    """
    def __init__(self, test: str, device: str, jit: bool=False, batch_size: Optional[int]=None, extra_args: List[str]=[]):
        self.metadata = self.load_metadata()
        self.test = test
        assert self.test == "train" or self.test == "eval", \
            f"Test must be 'train' or 'eval', but get {self.test}. Please submit a bug report."
        self.device = device
        self.jit = jit
        self.determine_batch_size(batch_size)
        self.extra_args = extra_args
        self.opt = None
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

        # taken from torchdynamo benchmarks, this further controls randomness settings
        def deterministic_torch_manual_seed(*args, **kwargs):
            from torch._C import default_generator

            seed = 1337
            import torch.cuda

            if not torch.cuda._is_in_bad_fork():
                torch.cuda.manual_seed_all(seed)

            return default_generator.manual_seed(seed)

        torch.manual_seed = deterministic_torch_manual_seed
        set_random_seed()
        # sanity checks of the options
        assert self.test == "train" or self.test == "eval", f"Test must be 'train' or 'eval', but provided {self.test}."
        # parse the args
        self.dargs, opt_args = parse_decoration_args(self, self.extra_args)
        # if the args contain "--torchdynamo", parse torchdynamo args
        if "--torchdynamo" in opt_args:
            self.dynamo = True
            from torchbenchmark.util.backends.torchdynamo import parse_torchdynamo_args
            self.opt_args, self.extra_args = parse_torchdynamo_args(self, opt_args)
        else:
            self.dynamo = False
            self.opt_args, self.extra_args = parse_opt_args(self, opt_args)

    # Run the post processing for model acceleration
    def __post__init__(self):
        # All arguments should be parsed at this point.
        assert not self.extra_args, f"Exptected no unknown args at this point, found {self.extra_args}"
        should_check_correctness = check_correctness_p(self, self.opt_args, self.dargs)
        if should_check_correctness:
            self.eager_output = stableness_check(self, cos_sim=False, deepcopy=self.DEEPCOPY, rounds=1)
            if isinstance(self.eager_output, Tuple):
                self.eager_output = tuple((t.detach() if isinstance(t, torch.Tensor) else t) for t in self.eager_output)
            elif isinstance(self.eager_output, torch.Tensor):
                self.eager_output = self.eager_output.detach()
            if self.test == "train":
                opt_saved = None
                if hasattr(self, "opt"):
                    opt_saved = self.opt
                    self.opt = None
                try:
                    if self.DEEPCOPY:
                        copy_model = copy.deepcopy(self)
                    else:
                        copy_model = self
                    copy_model.invoke()
                    self.eager_model_after_one_train_iteration = copy_model.model
                except RuntimeError:
                    warnings.warn(UserWarning("Can't copy the model. Skipping train correctness check."))
                if opt_saved:
                    self.opt = opt_saved
        # apply decoration args
        apply_decoration_args(self, self.dargs)
        # apply optimization args
        if self.dynamo:
            from torchbenchmark.util.backends.torchdynamo import apply_torchdynamo_args
            apply_torchdynamo_args(self, self.opt_args, self.dargs.precision)
        else:
            apply_opt_args(self, self.opt_args)
        if should_check_correctness:
            # tensorrt or fp16 is known to generate less-accurate results
            # in this case, use more relaxed cosine similarity instead of torch.allclose
            # for correctness testing
            # see: https://github.com/pytorch/torchdynamo/pull/438
            if (
                self.dargs.precision == "fp16"
                or self.dargs.precision == "amp"
                or (self.dynamo and self.opt_args.torchdynamo == "fx2trt")
                or (not self.dynamo and (self.device == "cuda" and self.opt_args.backend == "fx2trt"))
                or (not self.dynamo and self.opt_args.use_cosine_similarity)
                or self.dargs.precision == "fx_int8"
            ):
                self.correctness = correctness_check(self, cos_sim=True, deepcopy=self.DEEPCOPY)
            else:
                # get tolerance of correctness check from os.environ
                atol = float(os.environ.get("TORCHBENCH_ATOL", "1e-4"))
                rtol = float(os.environ.get("TORCHBENCH_RTOL", "1e-4"))
                self.correctness = correctness_check(self, cos_sim=False, deepcopy=self.DEEPCOPY, atol=atol, rtol=rtol)
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
        if self.test == "cuda":
            torch.cuda.empty_cache()

    def determine_batch_size(self, batch_size=None):
        # batch size priority for eval tests: not ALLOW_CUSTOMIZE_BSIZE > user specified > device specified > default
        # batch size priority for train tests: not ALLOW_CUSTOMIZE_BSIZE > user specified > default
        self.batch_size = batch_size
        if not batch_size:
            self.batch_size = self.DEFAULT_TRAIN_BSIZE if self.test == "train" else self.DEFAULT_EVAL_BSIZE
            # use the device suggestion on CUDA inference tests
            if self.test == "eval":
                if self.device == "cuda":
                    current_device_name = torch.cuda.get_device_name()
                    assert current_device_name, f"torch.cuda.get_device_name() returns None when device is set to cuda, please double check."
                elif self.device == "cpu":
                    current_device_name = "cpu"
                elif self.device == "mps":
                    current_device_name = "mps"

                if self.metadata and "devices" in self.metadata and current_device_name in self.metadata["devices"]:
                    self.batch_size = self.metadata["devices"][current_device_name]["eval_batch_size"]
            # If the model doesn't implement test or eval test
            # its DEFAULT_TRAIN_BSIZE or DEFAULT_EVAL_BSIZE will still be None
            if not self.batch_size:
                raise NotImplementedError(f"Test {self.test} is not implemented.")
        else:
            self.batch_size = batch_size
        # Check if specified batch size is supported by the model
        if hasattr(self, "ALLOW_CUSTOMIZE_BSIZE") and (not getattr(self, "ALLOW_CUSTOMIZE_BSIZE")):
            if self.test == "train" and (not self.batch_size == self.DEFAULT_TRAIN_BSIZE):
                raise NotImplementedError("Model doesn't support customizing batch size.")
            elif self.test == "eval" and (not self.batch_size == self.DEFAULT_EVAL_BSIZE):
                raise NotImplementedError("Model doesn't support customizing batch size.")

    def load_metadata(self):
        relative_path = self.__class__.__module__.split(".")
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

    # Default implementation for replacing the model
    def set_module(self, new_model):
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            self.model = new_model
        else:
            raise NotImplementedError("The instance variable 'model' does not exist or is not type 'torch.nn.Module', implement your own `set_module()` function.")

    def gen_inputs(self, num_batches: int=1) -> Tuple[Generator, Optional[int]]:
        """Generate a tuple of (iterator of model input, the size of the iterator).
           If size is None, the input is randomly generated and has infinite size."""
        raise NotImplementedError("Default input generation function is not implemented. "
                                  "Please submit an issue if you need input iterator implementation for the model.")

    def invoke_staged_train_test(self) -> None:
        if hasattr(self, "opt") and self.opt:
            self.opt.zero_grad()

        with nested(*self.forward_contexts):
            losses = self.forward()

        with nested(*self.backward_contexts):
            self.backward(losses)

        if hasattr(self, "opt") and self.opt:
            with nested(*self.optimizer_contexts):
                self.optimizer()

        return None

    def invoke(self) -> Optional[Tuple[torch.Tensor]]:
        out = None
        if self.test == "train" and is_staged_train_test(self):
            self.invoke_staged_train_test()
            return out
        with nested(*self.run_contexts):
            if self.test == "train":
                self.train()
            elif self.test == "eval":
                out = self.eval()
        return out

    def eval_in_nograd(self):
        return True

    def check_opt_vs_noopt_jit(self):
        if not self.jit:
            return

        model_name = inspect.getfile(self.__class__).split(os.sep)[-2]
        print(f"model_name={model_name} , {inspect.getfile(self.__class__)}")
        model_blacklist = [
            'demucs', # set up issue
            'yolov3', # set up issue
            'BERT_pytorch', # set up issue
            'moco', # set up issue
            'Super_SloMo', # results don't match, might be due to the way TE CUDA handles rand?
            'attention_is_all_you_need_pytorch', # results don't match, might be due to the way TE CUDA handles rand?
        ]

        if model_name in model_blacklist:
            warnings.warn(UserWarning(f"{model_name}.get_module() doesn't support `check_results` yet!"))
            return

        # if a model doesn't support `get_module`
        # we should let it throw and then
        # override `check_results` for that model
        try:
            model, inputs = self.get_module()
        except NotImplementedError:
            warnings.warn(UserWarning(f"{model_name}.get_module() doesn't support `check_results` yet!"))
            return

        def bench_allclose(a, b):
            if isinstance(a, torch.Tensor):
                assert(isinstance(b, torch.Tensor))
                assert(a.allclose(b))
            elif isinstance(a, tuple) or isinstance (b, list):
                assert(type(a) == type(b))
                assert(len(a) == len(b))
                for i in range(len(a)):
                    bench_allclose(a[i], b[i])
            else:
                raise RuntimeError("Encountered an supported type.\n" +
                    "Please add the type or override `bench_allclose`")


        try:
            opt = model(*inputs)
        except Exception as e:
            print(e)
            warnings.warn(UserWarning(f"{model_name}.eval() doesn't support `check_results` yet!"))
            return

        # disable optimizations and force a recompilation
        # to a baseline version
        fwd = model._c._get_method("forward")
        fwd._debug_flush_compilation_cache()
        torch._C._set_graph_executor_optimize(False)
        base = model(*inputs)
        torch._C._set_graph_executor_optimize(True)

        bench_allclose(base, opt)

    def enable_channels_last(self):
        model_name = self.name
        try:
            model, _ = self.get_module()
            model = model.to(memory_format=torch.channels_last)
        except RuntimeError:
            warnings.warn(UserWarning(f"{model_name} doesn't support `channels_last` yet!"))
            return
        self.set_module(model)
        def inputs_convert(example_inputs):
            if isinstance(example_inputs, torch.Tensor) and example_inputs.dim()==4:
                return example_inputs.to(memory_format=torch.channels_last)
            elif isinstance(example_inputs, (tuple, list, dict)):
                return tree_map(lambda x: inputs_convert(x), example_inputs)
            else:
                warnings.warn(UserWarning(f"{model_name} example inputs doesn't convert to `channels_last`!"))
                return example_inputs
        if hasattr(self, 'example_inputs'):
            self.example_inputs = inputs_convert(self.example_inputs)
        else:
            warnings.warn(UserWarning(f"{model_name} example inputs doesn't convert to `channels_last`!"))     

    def enable_fx_int8(self, quant_engine:str='x86'):
        from torch.ao.quantization import QuantWrapper, get_default_qconfig_mapping, get_default_qconfig_propagation_list
        from torch.ao.quantization.quantize_fx import _fuse_fx, prepare_fx, convert_fx
        torch.backends.quantized.engine = quant_engine
        qconfig_mapping = get_default_qconfig_mapping(quant_engine)

        def _append_attr(fx_module, module, fx_white_list=[]):
            import re
            fx_attr = dir(fx_module)
            org_attr = dir(module)
            ignore_match_patterns = [r"_", r"quant", r"dequant", r"weight",
                                    r"bias", r'activation_post_process']
            ignore_search_patterns = [r"_scale_", r"_zero_point_",
                                    r'_activation_post_process_']
            add_special_patterns = [r"_forward_hooks", r"_forward_pre_hooks", r"_backward_hooks"]
            attr_names = []
            for i in org_attr:
                if type(module) in fx_white_list and type(module) != torch.nn.Sequential \
                and any([re.search(p, i) for p in add_special_patterns]):
                    continue
                if any([re.search(p, i) for p in add_special_patterns]) \
                or (i not in fx_attr \
                    and not any([re.match(p, i) for p in ignore_match_patterns]) \
                    and not any([re.search(p, i) for p in ignore_search_patterns])) :
                    attr_names.append(i)
            for name in attr_names:
                attr = getattr(module, name, None)
                if isinstance(attr, torch.nn.Module) or \
                isinstance(attr, torch.quantization.qconfig.QConfig):
                    continue
                setattr(fx_module, name, attr)
            return fx_module

        def _get_sub_module(model, module_dict, prefix):
            import transformers
            fx_white_list = get_default_qconfig_propagation_list()
            ignore_list = []
            if is_hf_model:
                import transformers
                ignore_list.extend([transformers.models.gpt2.modeling_gpt2.GPT2Attention, transformers.models.t5.modeling_t5.T5DenseActDense])
            for name, module in model.named_children():
                quant_wrap_flag = False
                if type(module) in ignore_list:
                    continue
                op_name = prefix + "." + name if prefix != "" else name
                if op_name not in module_dict:
                    continue
                if type(module) in fx_white_list and type(module) != torch.nn.Sequential:
                    module = QuantWrapper(module)
                    quant_wrap_flag = True
                try:
                    graph_module = torch.fx.symbolic_trace(module)
                    if not quant_wrap_flag and str(module.get_submodule).count("\n") != str(graph_module.get_submodule).count("\n"):
                        continue
                    _fuse_fx(graph_module, False)
                    setattr(model, name, module)
                    sub_module_list.append(op_name)
                except:
                    _get_sub_module(module, module_dict, op_name)
            return model

        def _prepare_sub_module(sub_module_list, model, prefix):
            for name, module in model.named_children():
                op_name = prefix + '.' + name if prefix != '' else name
                if op_name in sub_module_list:
                    prepared_module = prepare_fx(module, qconfig_mapping, None)
                    _append_attr(prepared_module, module)
                    setattr(model, name, prepared_module)
                else:
                    _prepare_sub_module(sub_module_list, module, op_name)
            return model

        def _convert_sub_module(sub_module_list, model, prefix):
            for name, module in model.named_children():
                op_name = prefix + '.' + name if prefix != '' else name
                if op_name in sub_module_list:
                    convert_module = convert_fx(module)
                    setattr(model, name, convert_module)
                else:
                    _convert_sub_module(sub_module_list, module, op_name)
            return model

        try:
            model, _ = self.get_module()
            sub_module_list = []
            # Get sub modules
            model = _get_sub_module(model, dict(model.named_modules()), '')
            if not len(sub_module_list):
                warnings.warn(UserWarning(f"{self.name} doesn't have submodule can ben quantized!"))
            model = _prepare_sub_module(sub_module_list, model, '')
            self.set_module(model)
            # Calibration
            self.eval()
            model, _ = self.get_module()
            model = _convert_sub_module(sub_module_list, model, '')
            self.set_module(model)
        except Exception as e:
            print(e)
            warnings.warn(UserWarning(f"{self.name} doesn't support `fx_int8` yet!"))
            raise RuntimeError
