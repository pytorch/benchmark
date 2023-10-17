"""
Stable Diffusion XL model
It requires users to specify "HUGGINGFACE_AUTH_TOKEN" in environment variable
to authorize login and agree HuggingFace terms and conditions.
"""
from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceAuthMixin

import torch
from diffusers import DiffusionPipeline


class Model(BenchmarkModel, HuggingFaceAuthMixin):
    task = COMPUTER_VISION.GENERATION

    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False
    # Default eval precision on CUDA device is fp16
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    def __init__(self, test, device, batch_size=None, extra_args=[]):
        HuggingFaceAuthMixin.__init__(self)
        super().__init__(test=test, device=device,
                         batch_size=batch_size, extra_args=extra_args)
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = DiffusionPipeline.from_pretrained(model_id).to(device)
        random_input = torch.randn(1, 4, 256, 256).to(device)
        timestep = torch.tensor([1.0]).to(device)
        encoder_hidden_states = torch.randn(1, 1, 2048).to(device)
        added_cond_kwargs = {
            "text_embeds": torch.randn(1, 2560).to(device),  # Example tensor, adjust shape as needed
            "time_ids": torch.tensor([1]).to(device)  # Replace 'some_value' with the appropriate value or tensor shape for time_ids
        }

        self.args_tuple = (random_input, timestep, encoder_hidden_states, None, None, None, None, added_cond_kwargs)

    def enable_fp16_half(self):
        pass
    
    def get_module(self):
        return self.pipe.unet, self.args_tuple

    def train(self):
        raise NotImplementedError("Train is not implemented for the stable diffusion XL model.")

    def eval(self):
        image = self.pipe.unet(*self.args_tuple)
        return (image, )