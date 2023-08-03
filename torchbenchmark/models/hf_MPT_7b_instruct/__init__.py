from torchbenchmark.tasks import NLP
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceModel

class Model(HuggingFaceModel):
    task = NLP.LANGUAGE_MODELING
    # https://huggingface.co/mosaicml/mpt-7b
    DEFAULT_TRAIN_BSIZE = 4
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(name="hf_MPT_7b_instruct", test=test, device=device, batch_size=batch_size, extra_args=extra_args)

    def eval(self):
        super().eval()