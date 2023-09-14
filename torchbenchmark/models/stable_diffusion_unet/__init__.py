"""
HuggingFace Stable Diffusion model.
It requires users to specify "HUGGINGFACE_AUTH_TOKEN" in environment variable
to authorize login and agree HuggingFace terms and conditions.
"""
from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceAuthMixin

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler


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
        super().__init__(test=test, device=device,
                         batch_size=batch_size, extra_args=extra_args)
        model_id = "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
        self.example_inputs = "a photo of an astronaut riding a horse on mars"
        self.pipe.to(self.device)

    def enable_fp16_half(self):
        pass

    
    def get_module(self):
        random_input = torch.randn(1, 4, 128, 128).to(self.device)
        timestep = torch.tensor([1.0]).to(self.device)
        encoder_hidden_states = torch.randn(1, 1, 1024).to(self.device)
        return self.pipe.unet, [random_input, timestep, encoder_hidden_states]


    def train(self):
        raise NotImplementedError("Train test is not implemented for the stable diffusion model.")

    def eval(self):
        image = self.pipe(self.example_inputs)
        return (image, )
