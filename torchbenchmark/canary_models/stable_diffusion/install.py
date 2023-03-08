from torchbenchmark.util.framework.diffusers import install_diffusers
from diffusers import StableDiffusionPipeline
import torch

MODEL_NAME = "stabilityai/stable-diffusion-2"

def load_model_checkpoint():
    StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, safety_checker=None)

if __name__ == '__main__':
    install_diffusers()
    load_model_checkpoint()
