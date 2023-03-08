from torchbenchmark.util.framework.diffusers import install_diffusers
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch

MODEL_NAME = "timbrooks/instruct-pix2pix"

def load_model_checkpoint():
    StableDiffusionInstructPix2PixPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, safety_checker=None)

if __name__ == '__main__':
    install_diffusers()
