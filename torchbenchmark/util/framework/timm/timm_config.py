import torch.nn as nn
import dataclasses
from timm.optim import create_optimizer

@dataclasses.dataclass
class OptimizerOption:
    lr: float
    opt: str
    weight_decay: float
    momentum: float

class TimmConfig:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Configurations
        self.num_classes = self.model.num_classes
        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.target_shape = tuple()
        self.input_size = self.model.default_cfg["input_size"]
        # Default optimizer configurations borrowed from:
        # https://github.com/rwightman/pytorch-image-models/blob/779107b693010934ac87c8cecbeb65796e218488/timm/optim/optim_factory.py#L78
        opt_args = OptimizerOption(lr=1e-4, opt="sgd", weight_decay = 0.0001, momentum = 0.9)
        self.optimizer = create_optimizer(opt_args, self.model)
