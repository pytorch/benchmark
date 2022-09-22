from torchbenchmark.tasks import COMPUTER_VISION
from torchbenchmark.util.model import BenchmarkModel

import torch
from diffusers import StableDiffusionPipeline

from typing import Tuple

class Model(BenchmarkModel):
    task = COMPUTER_VISION.GENERATION

    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1
    ALLOW_CUSTOMIZE_BSIZE = False

    def __init__(self, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit,
                         batch_size=batch_size, extra_args=extra_args)
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
        self.model = pipe.to(self.device)
        self.prompt = ("a photo of an astronaut riding a horse on mars", )

    def get_module(self):
        return (self.model, self.prompt)

    def train(self):
        raise NotImplementedError("Train test is not implemented for stable diffusion.")

    def eval(self) -> Tuple[torch.Tensor]:
        with torch.inference_mode(), torch.autocast(self.device):
            image = self.model(*self.prompt).images[0]
        return (image, )
