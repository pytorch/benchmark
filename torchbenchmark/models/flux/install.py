import os
import warnings

import torch
from torchbenchmark.util.framework.diffusers import install_diffusers

MODEL_NAME = "black-forest-labs/FLUX.1-dev"


def load_model_checkpoint():
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, safety_checker=None
    )

    return pipe


if __name__ == "__main__":
    install_diffusers()
    if not "HUGGING_FACE_HUB_TOKEN" in os.environ:
        warnings.warn(
            "Make sure to set `HUGGINGFACE_HUB_TOKEN` so you can download weights"
        )
    else:
        load_model_checkpoint()
