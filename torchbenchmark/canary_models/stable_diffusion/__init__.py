"""
HuggingFace Stable Diffusion model.
It requires users to specify "HUGGINGFACE_AUTH_TOKEN" in environment variable
to authorize login and agree HuggingFace terms and conditions.
"""
from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.model import BenchmarkModel

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler


class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION

    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False
    # Default eval precision on CUDA device is fp16
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"


    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit,
                         batch_size=batch_size, extra_args=extra_args)
        assert self.dargs.precision == "fp16", f"Stable Diffusion model only supports fp16 precision."
        model_id = "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
        self.pipe.to(self.device)
        self.example_inputs = "a photo of an astronaut riding a horse on mars"

    def enable_fp16_half(self):
        pass

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        raise NotImplementedError("Train test is not implemented for the stable diffusion model.")

    def eval(self):
        image = self.pipe(self.example_inputs)
        return (image, )
