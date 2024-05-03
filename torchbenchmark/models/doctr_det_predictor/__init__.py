"Doctr detection model"
from doctr.models import ocr_predictor
import torch

# TorchBench imports
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from typing import Tuple


class Model(BenchmarkModel):
    task = COMPUTER_VISION.DETECTION
    DEFAULT_EVAL_BSIZE = 1
    CANNOT_SET_CUSTOM_OPTIMIZER = True

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )

        predictor = ocr_predictor(
            det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
        ).to(self.device)
        # Doctr detection model expects input (batch_size, 3, 1024, 1024)
        self.model = predictor.det_predictor.model
        self.example_inputs = torch.randn(self.batch_size, 3, 1024, 1024).to(
            self.device
        )
        if self.test == "eval":
            self.model.eval()

    def train(self):
        raise NotImplementedError("Train is not implemented for this model.")

    def get_module(self):
        return self.model, (self.example_inputs,)

    def eval(self) -> Tuple[torch.Tensor]:
        out = self.model(self.example_inputs, return_model_output=True)
        return (out["out_map"],)
