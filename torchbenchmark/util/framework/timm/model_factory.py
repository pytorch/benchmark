import torch
import typing
import timm
from torchbenchmark.util.model import BenchmarkModel
from .timm_config import TimmConfig

class TimmModel(BenchmarkModel):
    optimized_for_inference = True
    # To recognize this is a timm model
    TIMM_MODEL = True
    # These two variables should be defined by subclasses
    DEFAULT_TRAIN_BSIZE = None
    DEFAULT_EVAL_BSIZE = None

    def __init__(self, model_name, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        self.model = timm.create_model(model_name, pretrained=False, scriptable=True)
        self.cfg = TimmConfig(model = self.model, device = device)
        self.example_inputs = self._gen_input(self.batch_size)

        self.model.to(
            device=self.device
        )
        if test == "train":
            self.model.train()
        elif test == "eval":
            self.model.eval()

        if device == 'cuda':
            torch.cuda.empty_cache()

    def _gen_input(self, batch_size):
        return torch.randn((batch_size,) + self.cfg.input_size, device=self.device)

    def _gen_target(self, batch_size):
        return torch.empty(
            (batch_size,) + self.cfg.target_shape,
            device=self.device, dtype=torch.long).random_(self.cfg.num_classes)

    def _step_train(self):
        self.cfg.optimizer.zero_grad()
        output = self.model(self.example_inputs)
        if isinstance(output, tuple):
            output = output[0]
        target = self._gen_target(output.shape[0])
        self.cfg.loss(output, target).backward()
        self.cfg.optimizer.step()

    def _step_eval(self):
        output = self.model(self.example_inputs)
        return output

    def get_module(self):
        return self.model, (self.example_inputs,)

    def _train(self, niter=1):
        self.model.train()
        for _ in range(niter):
            self._step_train()

    def _eval(self, niter=1) -> typing.Tuple[torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            for _ in range(niter):
                out = self._step_eval()
        return (out, )
