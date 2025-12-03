import torch
import torch.distributed as dist

if dist.is_available():
    from torch.testing._internal.distributed.fake_pg import FakeStore
from torchbenchmark.tasks import NLP

from ...util.model import BenchmarkModel
from .model import get_window_size_blocks, GPT


class GPTWrapper(torch.nn.Module):
    def __init__(
        self,
        model: GPT,
        target_seq: torch.Tensor,
        sliding_window_num_blocks: torch.Tensor,
    ):
        super().__init__()
        self.model = model
        self.target_seq = target_seq
        self.sliding_window_num_blocks = sliding_window_num_blocks

    def forward(self, input_seq: torch.Tensor):
        return self.model(input_seq, self.target_seq, self.sliding_window_num_blocks)


class Model(BenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1

    def validate_environment(self):
        if not torch.cuda.is_available() or "cuda" not in self.device:
            return NotImplementedError("Model requires CUDA")

        if not torch.cuda.is_bf16_supported():
            return NotImplementedError("Model requires BF16")

        if not dist.is_available():
            return NotImplementedError(
                "Model requires PyTorch built with USE_DISTRIBUTED=1"
            )

        return None

    def __init__(self, test="train", device="cuda", batch_size=None, extra_args=[]):
        super().__init__(
            test=test,
            device=device,
            batch_size=batch_size,
            extra_args=extra_args,
        )

        error = self.validate_environment()
        if error:
            raise error

        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake",
            store=fake_store,
            rank=0,
            world_size=8,
        )
        vocab_size = 50257
        max_seq_len = 262144
        input_seq_len = 6144  # originally 49152, lower to run flex attention in eager
        _model: nn.Module = GPT(
            vocab_size,
            num_layers=12,
            num_heads=6,
            model_dim=768,
            max_seq_len=max_seq_len,
        ).cuda()
        for m in _model.modules():
            if isinstance(m, torch.nn.Embedding):
                m.bfloat16()
        dist.destroy_process_group()

        self.example_inputs = (
            torch.randint(
                0, vocab_size, (input_seq_len,), dtype=torch.int32, device="cuda"
            ),
        )
        _target_inputs = torch.randint(
            0, vocab_size, (input_seq_len,), dtype=torch.int64, device="cuda"
        )
        _window_size = get_window_size_blocks(0.5, device="cuda")  # use avg window size
        self.model = GPTWrapper(_model, _target_inputs, _window_size)

    def get_module(self):
        return self.model, self.example_inputs

    def train(self):
        # I would limit this to H100, but I don't think we can do that
        # See main.py to run
        raise NotImplementedError("Training is not supported in CI")

    def eval(self):
        # I would limit this to H100, but I don't think we can do that
        # See main.py to run
        raise NotImplementedError("Eval is not supported in CI")
