import torch
import timm.models.nfnet

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
from .config import TimmConfig
from torchbenchmark.util.framework.timm.extra_args import parse_args, apply_args

class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION
    optimized_for_inference = True

    # Original train batch size 128, hardware Nvidia rtx 3090
    # Source: https://gist.github.com/rwightman/bb59f9e245162cee0e38bd66bd8cd77f#file-bench_by_train-csv-L147
    # Eval batch size 256, hardware Nvidia rtx 3090
    # Source: https://github.com/rwightman/pytorch-image-models/blob/f7d210d759beb00a3d0834a3ce2d93f6e17f3d38/results/model_benchmark_amp_nchw_rtx3090.csv
    # Downscale to 128 to fit T4
    def __init__(self, test="eval", device=None, jit=False,
                 variant='dm_nfnet_f0',
                 eval_bs=128, train_bs=128, extra_args=[]):
        super().__init__()
        self.device = device
        self.jit = jit
        self.test = test
        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.model = timm.create_model(variant, pretrained=False, scriptable=True)
        self.cfg = TimmConfig(model = self.model, device = device)
        self.example_inputs = self._gen_input(train_bs)
        self.eval_example_inputs = self._gen_input(eval_bs)
        self.model.to(
            device=self.device
        )
        if device == 'cuda':
            torch.cuda.empty_cache()

        # instantiate another model for inference
        self.eval_model = timm.create_model(variant, pretrained=False, scriptable=True)
        self.eval_model.eval()
        self.eval_model.to(
            device=self.device
        )

        # process extra args
        self.args = parse_args(self, extra_args)
        apply_args(self, self.args)

        if jit:
            self.model = torch.jit.script(self.model)
            self.eval_model = torch.jit.script(self.eval_model)
            assert isinstance(self.eval_model, torch.jit.ScriptModule)
            self.eval_model = torch.jit.optimize_for_inference(self.eval_model)

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

    # vision models have another model
    # instance for inference that has
    # already been optimized for inference
    def set_eval(self):
        pass

    def _step_eval(self):
        output = self.eval_model(self.eval_example_inputs)

    def get_module(self):
        return self.model, (self.example_inputs,)

    def train(self, niter=1):
        if self.device == "cuda":
            raise NotImplementedError("Disable the train test because it causes CUDA OOM on Nvidia T4")
        self.model.train()
        for _ in range(niter):
            self._step_train()

    def eval(self, niter=1):
        self.eval_model.eval()
        with torch.no_grad():
            for _ in range(niter):
                self._step_eval()
