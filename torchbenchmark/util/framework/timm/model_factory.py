import torch
import random
import numpy as np
import typing
import torch.optim as optim
from torchbenchmark.util.model import BenchmarkModel

class TimmModel(BenchmarkModel):
    optimized_for_inference = True
    # To recognize this is a torchvision model
    TIMM_MODEL = True
    # These two variables should be defined by subclasses
    DEFAULT_TRAIN_BSIZE = None
    DEFAULT_EVAL_BSIZE = None

    def __init__(self, model_name, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        torch.manual_seed(1337)
        random.seed(1337)
        np.random.seed(1337)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
