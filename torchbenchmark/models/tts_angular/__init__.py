
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH
import torch
from typing import Tuple
from .angular_tts_main import TTSModel, SYNTHETIC_DATA

class Model(BenchmarkModel):
    task = SPEECH.SYNTHESIS
    # Original train batch size: 64
    # Source: https://github.com/mozilla/TTS/blob/master/TTS/speaker_encoder/config.json#L38
    DEFAULT_TRAIN_BSIZE = 64
    DEFAULT_EVAL_BSIZE = 64

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        if jit:
            raise NotImplementedError("tts-angular model does not support JIT.")
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

        self.model = TTSModel(device=self.device, batch_size=self.batch_size)
        self.model.model.to(self.device)

    def get_module(self):
        return self.model.model, [SYNTHETIC_DATA[0], ]

    def set_module(self, new_model):
        self.model.model = new_model

    def set_train(self):
        self.model.model.train()

    def _train(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        # the training process is not patched to use scripted models
        self.model.train(niter)

    def _eval(self, niter=1) -> Tuple[torch.Tensor]:
        if self.jit:
            raise NotImplementedError()
        for _ in range(niter):
            out = self.model.eval()
        return (out, )
