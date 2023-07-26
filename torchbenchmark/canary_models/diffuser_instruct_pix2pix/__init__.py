from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.framework.diffusers.model_factory import DiffuserModel

class Model(DiffuserModel):
    task = COMPUTER_VISION.GENERATION
    DEFAULT_TRAIN_BSIZE = 4
    DEFAULT_EVAL_BSIZE = 1
    # Default eval precision on CUDA device is fp16
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(name="timbrooks/instruct-pix2pix",
                         test=test, device=device,
                         batch_size=batch_size, extra_args=extra_args)
