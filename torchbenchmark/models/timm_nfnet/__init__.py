import torch
import timm.models.nfnet

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
from .nfnet import NFNetConfig

class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION

    def __init__(self, device=None, jit=False, variant='dm_nfnet_f0', precision='float32'):
        super().__init__()
        self.device = device
        self.jit = jit
        self.model = timm.create_model(variant, pretrained=False, scriptable=True)
        self.cfg = NFNetConfig(model = self.model, device = device, precision = precision)
        self.model.to(
            device=self.device,
            dtype=self.cfg.model_dtype
        )
        if jit:
            self.model = torch.jit.script(self.model)

    def _gen_target(self, batch_size):
        return torch.empty(
            (batch_size,) + self.cfg.target_shape,
            device=self.device, dtype=torch.long).random_(self.cfg.num_classes)
    
    def _step_train(self):
        self.cfg.optimizer.zero_grad()
        output = self.model(self.cfg.example_inputs)
        if isinstance(output, tuple):
            output = output[0]
        target = self._gen_target(output.shape[0])
        self.cfg.loss(output, target).backward()
        self.cfg.optimizer.step()

    def _step_eval(self):
        output = self.model(self.cfg.example_inputs)

    def get_module(self):
        return self.model, self.cfg.example_inputs

    def train(self, niter=1):
        for _ in range(niter):
            self._step_train()
        
    # TODO: use pretrained model weights, assuming the pretrained model is in .data/ dir
    def eval(self, niter=1):
        with torch.no_grad():
            for _ in range(niter):
                self._step_eval()

if __name__ == "__main__":
    for device in ['cpu', 'cuda']:
        for jit in [False, True]:
            print("Test config: device {}, JIT {}".format(device, jit))
            m = Model(device=device, jit=jit)
            m, example_inputs = m.get_module()
            m(example_inputs)
            m.train()
            m.eval()
