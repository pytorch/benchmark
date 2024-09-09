from torchbenchmark.tasks import NLP
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceModel


class Model(HuggingFaceModel):
    task = NLP.LANGUAGE_MODELING
    # DEFAULT_TRAIN_BSIZE not specified since we're not implementing a train test
    # DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(
            name="moondream",
            test=test,
            device=device,
            batch_size=batch_size,
            extra_args=extra_args,
        )

    def train(self):
        return NotImplementedError("Not implemented")

    def eval(self):
        super().eval()
