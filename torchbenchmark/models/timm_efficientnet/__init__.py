from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.framework.timm.model_factory import TimmModel


class Model(TimmModel):
    task = COMPUTER_VISION.CLASSIFICATION

    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 64

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            test=test,
            model_name="efficientnet_b0",
            device=device,
            batch_size=batch_size,
            extra_args=extra_args,
        )
