import torch
import torch.nn as nn
import dataclasses
from timm.optim import create_optimizer

def resolve_precision(precision: str):
    assert precision in ('amp', 'float16', 'bfloat16', 'float32')
    use_amp = False
    model_dtype = torch.float32
    data_dtype = torch.float32
    if precision == 'amp':
        use_amp = True
    elif precision == 'float16':
        model_dtype = torch.float16
        data_dtype = torch.float16
    elif precision == 'bfloat16':
        model_dtype = torch.bfloat16
        data_dtype = torch.bfloat16
    return use_amp, model_dtype, data_dtype

@dataclasses.dataclass
class OptimizerOption:
    lr: float
    opt: str
    weight_decay: float
    momentum: float

class NFNetConfig:
    def _init_input(self):
        self.example_inputs = torch.randn(
            (self.batch_size,) + self.input_size, device=self.device, dtype=self.data_dtype)

    def __init__(self, model, device, precision):
        self.model = model
        self.device = device
        self.use_amp, self.model_dtype, self.data_dtype = resolve_precision(precision)
        # Configurations
        self.batch_size = 2
        self.num_classes = self.model.num_classes
        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.target_shape = tuple()
        self.input_size = self.model.default_cfg["input_size"]
        self._init_input()
        opt_args = OptimizerOption(lr=1e-4, opt="sgd", weight_decay = 0.0001, momentum = 0.9)
        self.optimizer = create_optimizer(opt_args, self.model)
