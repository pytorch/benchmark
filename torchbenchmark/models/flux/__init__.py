import torch
from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceAuthMixin
from torchbenchmark.util.model import BenchmarkModel

from .install import load_model_checkpoint


class Model(BenchmarkModel, HuggingFaceAuthMixin):
    task = COMPUTER_VISION.GENERATION

    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False
    # Skip deepcopy because it will oom on A100 40GB
    DEEPCOPY = False
    # Default eval precision on CUDA device is fp16
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        HuggingFaceAuthMixin.__init__(self)
        super().__init__(
            test=test, device=device, batch_size=batch_size, extra_args=extra_args
        )
        self.pipe = load_model_checkpoint()
        self.example_inputs = {
            "prompt": "A cat holding a sign that says hello world",
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "num_inference_steps": 50,
            "max_sequence_length": 512,
            "generator": torch.Generator("cpu").manual_seed(0),
        }
        self.pipe.to(self.device)

    def enable_fp16(self):
        # This model uses fp16 by default
        # Make this function no-op.
        pass

    def get_module(self):
        # A common configuration:
        # - resolution = 1024x1024
        # - maximum sequence length = 512
        #
        # The easiest way to get these metadata is probably to run the pipeline
        # with the example inputs, and then breakpoint at the transformer module
        # forward and print out the input tensor metadata.
        inputs = {
            "hidden_states": torch.randn(1, 4096, 64, device=self.device, dtype=torch.bfloat16),
            "encoder_hidden_states": torch.randn(1, 512, 4096, device=self.device, dtype=torch.bfloat16),
            "pooled_projections": torch.randn(1, 768, device=self.device, dtype=torch.bfloat16),
            "img_ids": torch.ones(1, 512, 3, device=self.device, dtype=torch.bfloat16),
            "txt_ids": torch.ones(1, 4096, 3, device=self.device, dtype=torch.bfloat16),
            "timestep": torch.tensor([1.0], device=self.device, dtype=torch.bfloat16),
            "guidance": torch.tensor([1.0], device=self.device, dtype=torch.bfloat16),
        }

        return self.pipe.transformer, inputs

    def set_module(self, mod):
        self.pipe.transformer = mod

    def train(self):
        raise NotImplementedError(
            "Train test is not implemented for the stable diffusion model."
        )

    def eval(self):
        image = self.pipe(**self.example_inputs)
        return (image,)
