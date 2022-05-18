from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION
    # Train batch size: use the smallest example batch of 128 (assuming only 1 worker)
    # Source: https://arxiv.org/pdf/1404.5997.pdf
    DEFAULT_TRAIN_BSIZE = 128
    DEFAULT_EVAL_BSIZE = 128

    def __init__(self, test, device, jit, batch_size=None, extra_args=[]):
        super().__init__(model_name="alexnet", test=test, device=device, jit=jit,
                         batch_size=batch_size, extra_args=extra_args)
