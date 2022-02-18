
# Generated by gen_torchvision_benchmark.py
import torch
import torch.optim as optim
import torchvision.models as models
from torch.quantization import quantize_fx
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION


class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION
    # Train batch size: 32
    # Source: https://openreview.net/pdf?id=B1Yy1BxCZ
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 32

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.model = models.resnet50().to(self.device)
        self.example_inputs = (torch.randn((self.batch_size, 3, 224, 224)).to(self.device),)
        self.prep_qat_train()
        if self.test == "eval":
            self.prep_qat_eval()

    def prep_qat_train(self):
        qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('fbgemm')}
        self.model.train()
        self.model = quantize_fx.prepare_qat_fx(self.model, qconfig_dict)

    def get_module(self):
        return self.model, self.example_inputs

    def prep_qat_eval(self):
        self.model = quantize_fx.convert_fx(self.model)
        self.model.eval()

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

    def eval(self, niter=1):
        if self.device != 'cpu':
            raise NotImplementedError()
        model = self.model
        example_inputs = self.example_inputs
        example_inputs = example_inputs[0][0].unsqueeze(0)
        for i in range(niter):
            model(example_inputs)
