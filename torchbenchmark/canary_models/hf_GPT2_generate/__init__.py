from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceGenerationModel

class Model(HuggingFaceGenerationModel):
    def __init__(self, test, device, batch_size=None, extra_args=[]):
        super().__init__(name="hf_GPT2_generate", test=test, device=device, batch_size=batch_size, extra_args=extra_args)
