from torchbenchmark.util.framework.timm.model_factory import TimmModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TimmModel):
    task = COMPUTER_VISION.GENERATION

    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 32

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(test=test, model_name='vit_small_patch16_224', device=device,
                         batch_size=batch_size, extra_args=extra_args)
