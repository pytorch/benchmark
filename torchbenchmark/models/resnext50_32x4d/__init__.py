from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION
    DEFAULT_TRAIN_BSIZE = 8
    DEFAULT_EVAL_BSIZE = 8

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(model_name="resnext50_32x4d", test=test, device=device, jit=jit,
                         batch_size=batch_size, extra_args=extra_args)
