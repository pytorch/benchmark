from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION
from torchvision import models

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION
    # Train batch size: use the smallest example batch of 128 (assuming only 1 worker)
    # Source: https://arxiv.org/pdf/1404.5997.pdf
    DEFAULT_TRAIN_BSIZE = 128
    DEFAULT_EVAL_BSIZE = 128

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(model_name="alexnet", test=test, device=device,
                         batch_size=batch_size, weights=models.AlexNet_Weights.IMAGENET1K_V1,
                         extra_args=extra_args)
