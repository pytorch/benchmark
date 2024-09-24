import argparse

from typing import List, Optional

from torchbenchmark.util.triton_op import BenchmarkOperator, register_benchmark

from .hstu import get_test_inputs, RaggedHSTUAttn


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--max-seq-len-log2", type=int, default=9)
    parser.add_argument("--num-buckets", type=int, default=2048)
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args=extra_args)
        args = parse_op_args(self.extra_args)
        self.batch_size = args.batch_size
        self.num_heads = args.heads
        self.max_seq_len = 2**args.max_seq_len_log2
        self.num_buckets = args.num_buckets
        # set a default number of inputs
        self._num_inputs = 10

    @register_benchmark()
    def hstu_triton_ragged_attention(self, qkv, seq_offsets, timestamps):
        attn = RaggedHSTUAttn(
            self.batch_size,
            self.num_heads,
            self.max_seq_len,
            self.num_buckets,
            persistent_kernel=False,
        )
        return lambda: attn(qkv, seq_offsets, timestamps)

    @register_benchmark()
    def hstu_triton_ragged_attention_persistent(self, qkv, seq_offsets, timestamps):
        attn = RaggedHSTUAttn(
            self.batch_size,
            self.num_heads,
            self.max_seq_len,
            self.num_buckets,
            persistent_kernel=True,
        )
        return lambda: attn(qkv, seq_offsets, timestamps)

    def get_x_val(self, example_inputs):
        return (self.batch_size, self.num_heads, self.max_seq_len, self.num_buckets)

    def get_input_iter(self):
        for _input_id in range(self._num_inputs):
            inputs = get_test_inputs(self.batch_size, self.num_heads, self.max_seq_len)
            yield inputs
