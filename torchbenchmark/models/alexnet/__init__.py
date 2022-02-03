from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Train batch size: use the smallest example batch of 128 (assuming only 1 worker)
    # Source: https://arxiv.org/pdf/1404.5997.pdf
    def __init__(self, test="eval", device=None, jit=False, train_bs=128, eval_bs=16, extra_args=[]):
        super().__init__(model_name="alexnet", test="eval", device=device, jit=jit,
                         train_bs=train_bs, eval_bs=eval_bs, extra_args=extra_args)

