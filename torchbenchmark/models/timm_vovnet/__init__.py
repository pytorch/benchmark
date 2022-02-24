import torch
import timm.models.vovnet

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
from .config import TimmConfig

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION
    optimized_for_inference = True

    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 32

    def __init__(self, test, device, jit=False, variant='vovnet39a',
                 batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.model = timm.create_model(variant, pretrained=False, scriptable=True)
        self.cfg = TimmConfig(model = self.model, device = device)
        self.example_inputs = self._gen_input(self.batch_size)
        self.model.to(
            device=self.device
        )
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

    def set_eval(self):
        self.model.eval()

    def _step_eval(self):
        output = self.model(self.example_inputs)

    def get_module(self):
        return self.model, (self.example_inputs,)

    def _train(self, niter=1):
        self.model.train()
        for _ in range(niter):
            self._step_train()

    def _eval(self, niter=1):
        self.model.eval()
        with torch.no_grad():
            for _ in range(niter):
                self._step_eval()

