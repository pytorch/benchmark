from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION
    optimized_for_inference = True

    def __init__(self, device=None, jit=False, train_bs=32, eval_bs=32, extra_args=[]):
        super().__init__(model_name="resnet50", device=device, jit=jit,
                         train_bs=train_bs, eval_bs=eval_bs, extra_args=extra_args)
