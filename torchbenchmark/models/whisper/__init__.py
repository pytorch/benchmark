import io
import torch
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import SPEECH
from datasets import load_dataset
# from model import Whisper, ModelDimensions
from torchbenchmark.models.whisper.install import load_model
from torchbenchmark.models.whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
from torchbenchmark.models.whisper.decoding import DecodingOptions, decode

NUM_TRAIN_BATCH = 1
NUM_EVAL_BATCH = 1

class Model(BenchmarkModel):
    task = SPEECH.RECOGNITION
    DEFAULT_TRAIN_BSIZE = 32
    DEFAULT_EVAL_BSIZE = 32

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        # Failing on cpu and batch sizes that are too large
        if self.device == 'cpu':
            return NotImplementedError("CPU test too slow - skipping.")
        if batch_size > 72:
            error_msg = """
                Batch sizes over 72 not presently supported.
            """
            return NotImplementedError(error_msg)
        self.model = load_model("medium", self.device, "./.data", in_memory=True)
        # Importing dataset and preprocessing
        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        mels=[]
        for i in range(self.batch_size):
            sample = dataset[i]["audio"]["path"]
            input_audio = load_audio(sample)
            trimmed_audio = pad_or_trim(input_audio)
            mels.append(log_mel_spectrogram(trimmed_audio))
        mels = torch.stack(mels)
        self.example_inputs = mels.to(self.device)
        self.model_args = DecodingOptions(fp16=False)

        
    def get_module(self):
        return self.model, self.example_inputs
    
    def train(self):
        error_msg = """
            Training not implemented.
        """
        return NotImplementedError(error_msg)

    def eval(self):
        with torch.no_grad():
            return self.model.decode(self.example_inputs, self.model_args)