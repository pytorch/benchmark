import torch
import timm.models.nfnet

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
from .config import TimmConfig

class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION
    optimized_for_inference = True

    # Original train batch size 128, hardware Nvidia rtx 3090
    # Source: https://gist.github.com/rwightman/bb59f9e245162cee0e38bd66bd8cd77f#file-bench_by_train-csv-L147
    # Eval batch size 256, hardware Nvidia rtx 3090
    # Source: https://github.com/rwightman/pytorch-image-models/blob/f7d210d759beb00a3d0834a3ce2d93f6e17f3d38/results/model_benchmark_amp_nchw_rtx3090.csv
    # Downscale to 128 to fit T4
    DEFAULT_TRAIN_BSIZE = 128
    DEFAULT_EVAL_BSIZE = 128

    def __init__(self, test, device, jit=False,
                 variant='dm_nfnet_f0',
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

        if jit:
            self.model = torch.jit.script(self.model)
            assert isinstance(self.model, torch.jit.ScriptModule)
            if test == "eval":
                self.model = torch.jit.optimize_for_inference(self.model)

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

    def train(self, niter=1):
        if self.device == "cuda":
            raise NotImplementedError("Disable the train test because it causes CUDA OOM on Nvidia T4")
        self.model.train()
        for _ in range(niter):
            self._step_train()

    def eval(self, niter=1):
        self.model.eval()
        with torch.no_grad():
            for _ in range(niter):
                self._step_eval()
