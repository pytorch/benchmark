from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION
import torchvision.models as models

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION
    # Train batch size: use the training batch in paper.
    # Source: https://arxiv.org/pdf/1608.06993.pdf
    DEFAULT_TRAIN_BSIZE = 256
    DEFAULT_EVAL_BSIZE = 64

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(model_name="densenet121", test=test, device=device, jit=jit,
                         batch_size=batch_size, weights=models.DenseNet121_Weights.IMAGENET1K_V1,
                         extra_args=extra_args)
