
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH

from .angular_tts_main import TTSModel, EVAL_SYNTHETIC_DATA

class Model(BenchmarkModel):
    task = SPEECH.SYNTHESIS
    # Original train batch size: 64
    # Source: https://github.com/mozilla/TTS/blob/master/TTS/speaker_encoder/config.json#L38
    def __init__(self, test="eval", device=None, jit=False, train_bs=64, eval_bs=256):
        super().__init__()
        self.device = device
        self.jit = jit
        self.model = TTSModel(device=self.device, train_bs=train_bs, eval_bs=eval_bs)
        self.model.model.to(self.device)

    def get_module(self):
        return self.model.model, [EVAL_SYNTHETIC_DATA[0], ]

    def set_train(self):
        # another model instance is used for training
        # and the train mode is on by default
        pass

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        # the training process is not patched to use scripted models
        self.model.train(niter)

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        for _ in range(niter):
            self.model.eval()
