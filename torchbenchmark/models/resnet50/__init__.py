from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchvision import models


class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 32

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            model_name="resnet50",
            test=test,
            device=device,
            batch_size=batch_size,
            weights=models.ResNet50_Weights.IMAGENET1K_V1,
            extra_args=extra_args,
        )
