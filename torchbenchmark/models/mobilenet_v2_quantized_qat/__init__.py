# Generated by gen_torchvision_benchmark.py
import torch
import torch.optim as optim
import torchvision.models as models
from torch.quantization import quantize_fx
from ...util.model import BenchmarkModel


class Model(BenchmarkModel):
    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit
        self.model = models.mobilenet_v2().to(self.device)
        self.example_inputs = (torch.randn((96, 3, 224, 224)).to(self.device),)
        self.prep_qat_train()  # config+prepare steps are required for both train and eval
        torch.cuda.empty_cache()

    def prep_qat_train(self):
        qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('fbgemm')}
        self.model.train()
        self.model = quantize_fx.prepare_qat_fx(self.model, qconfig_dict)

    def train(self, niter=3):
        if self.jit is True:  # torchscript operations should only be applied after quantization operations
            raise NotImplementedError()
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            optimizer.zero_grad()
            pred = self.model(*self.example_inputs)
            y = torch.empty(pred.shape[0], dtype=torch.long, device=self.device).random_(pred.shape[1])
            loss(pred, y).backward()
            optimizer.step()

    def set_eval(self):
        self.prep_qat_eval()

    def prep_qat_eval(self):
        self.model = quantize_fx.convert_fx(self.model)
        if self.jit:
            self.model = torch.jit.script(self.model)
        self.model.eval()

    def eval(self, niter=1):
        if self.device != 'cpu':
            raise NotImplementedError()
        model, example_inputs = self.get_module()
        example_inputs = example_inputs[0][0].unsqueeze(0)
        for _i in range(niter):
            model(example_inputs)

    def get_module(self):
        return self.model, self.example_inputs


if __name__ == "__main__":
    m = Model(device="cuda", jit=True)
    module, example_inputs = m.get_module()
    module(*example_inputs)
    m.eval(niter=1)
    m.train(niter=1)
