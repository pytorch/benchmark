default_operators = [
    "addmm",
    "bf16xint16_gemm",
    "flash_attention",
    "fp8_attention",
    "fp8_fused_quant_gemm_rowwise",
    "fp8_gemm",
    "fp8_gemm_blockwise",
    "fp8_gemm_rowwise",
    "gather_gemv",
    "gemm",
    "grouped_gemm",
    "int4_gemm",
    "jagged_layer_norm",
    "jagged_mean",
    "jagged_softmax",
    "jagged_sum",
    "launch_latency",
    "layer_norm",
    "low_mem_dropout",
    "ragged_attention",
    "softmax",
    "sum",
    "template_attention",
    "test_op",
    "vector_add",
    "welford",
]


def get_operators():
    return default_operators
