from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION
    # Train batch size: use the training batch in paper.
    # Source: https://arxiv.org/pdf/1608.06993.pdf
    DEFAULT_TRAIN_BSIZE = 256
    DEFAULT_EVAL_BSIZE = 64

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        # Temporarily disable tests because it causes CUDA OOM on CI platform
        # TODO: Re-enable these tests when better hardware is available
        if device == 'cuda':
            raise NotImplementedError('CUDA disabled due to CUDA out of memory on CI GPU')
        if device == 'cpu':
            raise NotImplementedError('CPU disabled due to out of memory on CI CPU')
        super().__init__(model_name="densenet121", test=test, device=device, jit=jit,
                         batch_size=batch_size, extra_args=extra_args)
