from torchbenchmark.util.framework.diffusers import install_diffusers
from torchbenchmark.util.framework.huggingface.model_factory import HuggingFaceAuthMixin
import torch
import os

MODEL_NAME = "stabilityai/stable-diffusion-2"

def load_model_checkpoint():
    from diffusers import StableDiffusionPipeline                   │
    StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, safety_checker=None)

def main():
    if not 'HUGGING_FACE_HUB_TOKEN' in os.environ:
        return NotImplementedError("Make sure to set `HUGGINGFACE_HUB_TOKEN` so you can download weights")
    else:
        install_diffusers()
        load_model_checkpoint()

if __name__ == "__main__":
    main()
