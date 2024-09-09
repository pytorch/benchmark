import os

from torchbenchmark.util.framework.huggingface.patch_hf import (
    cache_model,
    patch_transformers,
)

if __name__ == "__main__":
    model_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    cache_model(model_name)
