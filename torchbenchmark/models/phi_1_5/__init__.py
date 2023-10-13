from torchbenchmark.tasks import NLP
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceModel

class Model(HuggingFaceModel):
    task = NLP.LANGUAGE_MODELING
    # TODO: is this needed given that this is a pre-trained model?
    # DEFAULT_TRAIN_BSIZE = 6
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(name="phi_1_5", test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

    def eval(self):
        super().eval()