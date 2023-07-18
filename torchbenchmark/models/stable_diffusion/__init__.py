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
    # Default eval precision on CUDA device is fp16
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        HuggingFaceAuthMixin.__init__(self)
        super().__init__(test=test, device=device, jit=jit,
                         batch_size=batch_size, extra_args=extra_args)
        model_id = "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
        self.pipe.to(self.device)
        self.example_inputs = "a photo of an astronaut riding a horse on mars"

    def enable_fp16_half(self):
        pass

    
    def get_module(self):
        batch_size = 1
        sequence_length = 10
        vocab_size = 32000

        # Generate random indices within the valid range
        input_tensor = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))

        # Make sure the tensor has the correct data type
        input_tensor = input_tensor.long().to(self.device)
        return self.pipe.text_encoder, [input_tensor]


    def train(self):
        raise NotImplementedError("Train test is not implemented for the stable diffusion model.")

    def eval(self):
        image = self.pipe(self.example_inputs)
        return (image, )
