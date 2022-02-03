from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Train batch size: use the training batch in paper.
    # Source: https://arxiv.org/pdf/1608.06993.pdf
    def __init__(self, test="eval", device=None, jit=False, train_bs=256, eval_bs=64, extra_args=[]):
        # Temporarily disable tests because it causes CUDA OOM on CI platform
        # TODO: Re-enable these tests when better hardware is available
        if device == 'cuda':
            raise NotImplementedError('CUDA disabled due to CUDA out of memory on CI GPU')
        if device == 'cpu':
            raise NotImplementedError('CPU disabled due to out of memory on CI CPU')
        super().__init__(model_name="densenet121", test="eval", device=device, jit=jit,
                         train_bs=train_bs, eval_bs=eval_bs, extra_args=extra_args)
