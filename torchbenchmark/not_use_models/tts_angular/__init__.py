
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH
import torch
from typing import Tuple
from .angular_tts_main import TTSModel

class Model(BenchmarkModel):
    task = SPEECH.SYNTHESIS
    # Original train batch size: 64
    # Source: https://github.com/mozilla/TTS/blob/master/TTS/speaker_encoder/config.json#L38
    DEFAULT_TRAIN_BSIZE = 64
    DEFAULT_EVAL_BSIZE = 64

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.model = TTSModel(device=self.device, batch_size=self.batch_size)
        self.model.model.to(self.device)
        if self.test == "train":
            self.model.model.train()
        elif self.test == "eval":
            self.model.model.eval()

    def get_module(self):
        return self.model.model, [self.model.SYNTHETIC_DATA[0], ]

    def set_module(self, new_model):
        self.model.model = new_model

    def train(self):
        # the training process is not patched to use scripted models
        self.model.train()

    def eval(self) -> Tuple[torch.Tensor]:
        out = self.model.eval()
        return (out, )
