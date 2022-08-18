import torch
from typing import Optional, List
from contextlib import contextmanager, ExitStack
from typing import ContextManager

class PostInitProcessor(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post__init__()
        return obj

@contextmanager
def nested(*contexts):
    """
    Chain and apply a list of contexts
    """
    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield contexts

class E2EBenchmarkModel(metaclass=PostInitProcessor):
    """
    A base class for adding models for all e2e models.
    """
    def __init__(self, test: str, batch_size: Optional[int]=None, extra_args: List[str]=[]):
        self.test = test
        assert self.test == "train" or self.test == "eval", f"Test must be 'train' or 'eval', but get {self.test}. Please submit a bug report."
        self.batch_size = batch_size
        if not self.batch_size:
            self.batch_size = self.DEFAULT_TRAIN_BSIZE if test == "train" else self.DEFAULT_EVAL_BSIZE
            # If the model doesn't implement test or eval test
            # its DEFAULT_TRAIN_BSIZE or DEFAULT_EVAL_BSIZE will still be None
            if not self.batch_size:
                raise NotImplementedError(f"Test {test} is not implemented.")
        self.extra_args = extra_args
        if "--torchdynamo" in self.extra_args:
            self.dynamo = True
            from torchbenchmark.util.backends.torchdynamo import parse_torchdynamo_args
            self.opt_args, self.extra_args = parse_torchdynamo_args(self, self.extra_args)
        else:
            self.dynamo = False

    # Run the post processing for model acceleration
    def __post__init__(self):
        # sanity checks of the options
        assert self.test == "train" or self.test == "eval", f"Test must be 'train' or 'eval', but provided {self.test}."
        # initialize run contexts
        self.run_contexts = []
        if self.dynamo:
            from torchbenchmark.util.backends.torchdynamo import apply_torchdynamo_args
            apply_torchdynamo_args(self, self.opt_args, precision=self.tb_args.fp16)

    def add_context(self, context_fn):
        ctx = context_fn()
        assert isinstance(ctx, ContextManager), f"Expected adding a ContextManager, get {type(ctx)}. Please report a bug."
        self.run_contexts.append(context_fn)

    def next_batch(self):
        raise NotImplementedError("Every E2EModel should implement this")
    
    def run_forward(self, input):
        raise NotImplementedError("Every E2EModel should implement this")

    def run_backward(self, loss):
        raise NotImplementedError("Every E2EModel should implement this")

    def run_optimizer_step(self):
        raise NotImplementedError("Every E2EModel should implement this")
