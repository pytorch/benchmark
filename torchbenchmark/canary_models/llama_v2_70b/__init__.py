from torchbenchmark.tasks import NLP
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceModel, HuggingFaceAuthMixin

class Model(HuggingFaceModel, HuggingFaceAuthMixin):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    DEEPCOPY = False 

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        HuggingFaceAuthMixin.__init__(self)
        super().__init__(name="llama_v2_70b", test=test, device=device, batch_size=batch_size, extra_args=extra_args)

  
    def train(self):
        return NotImplementedError("FSDP should implement a training loop")
