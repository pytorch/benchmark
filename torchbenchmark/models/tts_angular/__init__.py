
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION

from .angular_tts_main import TTSModel, SYNTHETIC_DATA

class Model(BenchmarkModel):
    task = SPEECH.SYNTHESIS
    def __init__(self, device=None, jit=False):
        super().__init__()
        self.device = device
        self.jit = jit
        self.model = TTSModel(device=self.device)

    def get_module(self):
        return self.model.model, [SYNTHETIC_DATA[0], ]

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


if __name__ == '__main__':
    m = Model(device='cpu', jit=False)
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()
