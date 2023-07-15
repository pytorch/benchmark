from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceModel
from torchbenchmark.tasks import SPEECH
import torch

class Model(HuggingFaceModel):
    task = SPEECH.RECOGNITION
    # https://cdn.openai.com/papers/whisper.pdf Says for large-v2 they trained on 1024 batch sizes, with 16 GPUs
    DEFAULT_EVAL_BSIZE = 64
    
    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(name="hf_Whisper", test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        self.feature_size = 80
        self.sequence_length = 3000
        input_features = torch.randn(size=(self.batch_size, self.feature_size, self.sequence_length),device=self.device)
        self.example_inputs = {"input_features": input_features.to(self.device)}

    def eval(self):
        super().eval()
    def train(self):
        raise NotImplementedError("Training is not implemented.")