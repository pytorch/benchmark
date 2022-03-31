from torchbenchmark.util.framework.timm.model_factory import TimmModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TimmModel):
    task = COMPUTER_VISION.CLASSIFICATION

    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 64

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, model_name='efficientnet_b0', device=device,
                         jit=jit, batch_size=batch_size, extra_args=extra_args)
