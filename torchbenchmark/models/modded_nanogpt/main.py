import torch

from . import Model


# python -m torchbenchmark.models.modded_nanogpt.main
if __name__ == "__main__":
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    benchmark_model = Model("train", "cuda")
    model, inputs = benchmark_model.get_module()
    del benchmark_model
    model = torch.compile(model, dynamic=False, fullgraph=True)

    loss = model(*inputs)
    loss.backward()
    print(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
