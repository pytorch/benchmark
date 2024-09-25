from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchvision import models


class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION
    DEFAULT_TRAIN_BSIZE = 128
    DEFAULT_EVAL_BSIZE = 64

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            model_name="shufflenet_v2_x1_0",
            test=test,
            device=device,
            batch_size=batch_size,
            weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
            extra_args=extra_args,
        )
