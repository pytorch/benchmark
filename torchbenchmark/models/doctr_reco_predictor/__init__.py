"Doctr recognition model"
from doctr.models import ocr_predictor
import numpy as np
import torch

# TorchBench imports
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from typing import Tuple

class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        predictor = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True).to(self.device)
        # recognition model expects input (batch_size, 3, 32, 128)
        self.model = predictor.reco_predictor.model
        # fake document file
        self.example_inputs = (torch.randn(self.batch_size, 3, 32, 128), )
        if self.test == "eval":
            self.model.eval()

    def train(self):
        raise NotImplementedError("Train is not implemented for this model.")

    def get_module(self):
        return self.model, self.example_inputs

    def eval(self) -> Tuple[torch.Tensor]:
        with torch.inference_mode():
            out = self.model(*self.example_inputs)
        return out
