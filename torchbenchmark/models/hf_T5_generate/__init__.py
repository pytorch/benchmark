from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceGenerationModel

class Model(HuggingFaceGenerationModel):
    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(name="hf_T5_generate", test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
