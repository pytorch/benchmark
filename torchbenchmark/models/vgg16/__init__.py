from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Original train batch size 256 on 4-GPU system
    # Downscale to 64 to run on single GPU device
    # Source: https://arxiv.org/pdf/1409.1556.pdf
    def __init__(self, test, device, jit=False, train_bs=64, eval_bs=4, extra_args=[]):
        super().__init__(model_name="vgg16", test=test, device=device, jit=jit,
                         train_bs=train_bs, eval_bs=eval_bs, extra_args=extra_args)
