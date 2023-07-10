from torchbenchmark.tasks import NLP
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceModel

class Model(HuggingFaceModel):
    task = NLP.LANGUAGE_MODELING
    # Published training batch size is 2304: see https://huggingface.co/tiiuae/falcon-7b/blob/main/README.md
    DEFAULT_TRAIN_BSIZE = 2304
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(name="hf_Falcon_7b", test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)

    def eval(self):
        if (self.device == "cpu"):
            raise NotImplementedError("Falcon model is too slow on CPU - skip CPU test.")
        super().eval()
