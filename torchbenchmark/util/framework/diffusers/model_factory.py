import torch
from torchbenchmark.util.model import BenchmarkModel
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from typing import Optional, List


class DiffuserModel(BenchmarkModel):
    DIFFUSER_MODEL = True

    def __init__(self, name: str, test: str, device: str, jit: bool = False, batch_size: Optional[int] = None, extra_args: List[str] = ...):
        super().__init__(test, device, jit, batch_size, extra_args)
        if self.device == "cpu":
            raise NotImplementedError(f"Model {self.name} does not support CPU device.")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(name, torch_dtype=torch.float32, safety_checker=None)
        pipe.to(self.device)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe
        # use the same size as the example image
        # https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg
        self.example_inputs = torch.randn(self.batch_size, 3, 128, 128).to(self.device)

    def train(self):
        raise NotImplementedError(f"Train is not implemented for model {self.name}")

    def eval(self):
        images = self.pipe("turn him into cyborg", image=self.example_inputs).images