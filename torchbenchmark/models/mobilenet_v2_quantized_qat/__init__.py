# Generated by gen_torchvision_benchmark.py
import torch
import torch.optim as optim
import torchvision.models as models
from torch.quantization import quantize_fx
from torchbenchmark.tasks import COMPUTER_VISION
from ...util.model import BenchmarkModel
from typing import Tuple


class Model(BenchmarkModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Train batch size: 96
    # Source: https://arxiv.org/pdf/1801.04381.pdf
    DEFAULT_TRAIN_BSIZE = 96
    DEFAULT_EVAL_BSIZE = 96

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        if test == "eval" and device != "cpu":
            raise NotImplementedError("The eval test only supports CPU.")
        if jit and test == "train":
            raise NotImplementedError("torchscript operations should only be applied after quantization operations")
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.model = models.mobilenet_v2().to(self.device)
        self.example_inputs = (torch.randn((self.batch_size, 3, 224, 224)).to(self.device),)
        self.prep_qat_train()  # config+prepare steps are required for both train and eval
        if self.test == "eval":
            self.prep_qat_eval()

    def prep_qat_train(self):
        qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('fbgemm')}
        self.model.train()
        self.model = quantize_fx.prepare_qat_fx(self.model, qconfig_dict)

    def _train(self, niter=3):
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            optimizer.zero_grad()
            pred = self.model(*self.example_inputs)
            y = torch.empty(pred.shape[0], dtype=torch.long, device=self.device).random_(pred.shape[1])
            loss(pred, y).backward()
            optimizer.step()

    def prep_qat_eval(self):
        self.model = quantize_fx.convert_fx(self.model)
        self.model.eval()

    def _eval(self, niter=1) -> Tuple[torch.Tensor]:
        example_inputs = self.example_inputs[0][0].unsqueeze(0)
        for _i in range(niter):
            out = self.model(example_inputs)
        return (out, )

    def get_module(self):
        return self.model, self.example_inputs
