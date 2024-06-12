import itertools

import torch
import triton
import triton.language as tl


@triton.jit
def triton_sum_kernel_scalar_result(
    input_ptr,  # pointer to input matrix
    output_ptr,  # pointer to output matrix
    M,  # number of elements
    BLOCK_SIZE_M: tl.constexpr,  # number of elements per block
):
    pid = tl.program_id(axis=0)  # i-th block of input

    block_start = pid * BLOCK_SIZE_M
    # offsets have shape equal to input shape
    offsets = block_start + tl.arange(
        0, BLOCK_SIZE_M
    )  # create 1D vector (input shape) ranging from beginning to end of this program's block

    # mask has shape equal to input shape
    mask = offsets < M  # mask out offsets that are out of bounds for input

    # loaded pointers have shape equal to input shape
    x = tl.load(
        input_ptr + offsets, mask=mask, other=mask
    )  # load input, where the loaded pointers are in the desired input shape

    output = tl.sum(x)

    # output_offsets have shape equal to output shape
    output_offsets = tl.arange(
        0, 1
    )  # create offsets for scalar output pointer (output shape == (1,))

    # stored pointers have shape equal to output shape
    tl.store(
        output_ptr + output_offsets, output
    )  # store output, where the stored pointers are in the desired output shape


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_NON_REDUCE_DIM": b},
            num_warps=w,
        )
        for b, w in itertools.product(
            [2, 4, 8, 16], [2, 4, 8]  # block sizes  # number of warps
        )
    ],
    key=["M", "N"],
)
@triton.jit
def triton_sum_kernel_1D_result(
    input_ptr,  # pointer to input matrix
    output_ptr,  # pointer to output matrix
    # matrix dimensions (input)
    M,  # number of rows
    N,  # number of columns
    # block sizes (input)
    BLOCK_SIZE_NON_REDUCE_DIM: tl.constexpr,  # number of elements in non-reduction dimension per block
    BLOCK_SIZE_REDUCE_DIM: tl.constexpr,  # number of elements in reduction dimension per block
    # reduction dimension
    dim: tl.constexpr,  # dimension along which to sum
):
    pid = tl.program_id(axis=0)  # i-th block of input

    block_start_m, block_start_n = 0, 0
    offsets_m, offsets_n = None, None
    if dim == 0:
        block_start_n = pid * BLOCK_SIZE_REDUCE_DIM
        # offsets have shape equal to input shape
        offsets_m = block_start_m + tl.arange(
            0, BLOCK_SIZE_REDUCE_DIM
        )  # create 1D vector for offsets on M-th dimension
        offsets_n = block_start_n + tl.arange(
            0, BLOCK_SIZE_NON_REDUCE_DIM
        )  # create 1D vector for offsets on N-th dimension
    elif dim == 1:
        block_start_m = pid * BLOCK_SIZE_REDUCE_DIM
        # offsets have shape equal to input shape
        offsets_m = block_start_m + tl.arange(
            0, BLOCK_SIZE_NON_REDUCE_DIM
        )  # create 1D vector for offsets on M-th dimension
        offsets_n = block_start_n + tl.arange(
            0, BLOCK_SIZE_REDUCE_DIM
        )  # create 1D vector for offsets on N-th dimension

    # mask has shape equal to input shape
    mask_m = offsets_m < M
    mask_n = offsets_n < N

    # create 2D matrices of pointers and masks, using above M and N vectors
    idxs = (offsets_m[:, None] * N) + offsets_n
    mask = mask_m[:, None] & mask_n

    # loaded pointers have shape equal to input shape
    input = tl.load(
        input_ptr + idxs, mask=mask, other=mask
    )  # other=mask zeros out masked values from input

    output = tl.sum(input, axis=dim)

    # stored pointers have shape equal to output shape
    if dim == 0:  # store output along N-th dimension
        tl.store(output_ptr + offsets_n, output, mask=mask_n)
    elif dim == 1:  # store output along M-th dimension
        tl.store(output_ptr + offsets_m, output, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_K": b},
            num_warps=w,
        )
        for b, w in itertools.product(
            [2, 4, 16, 32, 128, 256], [2, 4, 8]  # block sizes  # number of warps
        )
    ],
    key=["N"],
)
@triton.jit
def triton_sum_kernel_2D_result_dim_1(
    input_ptr,  # pointer to input matrix
    output_ptr,  # pointer to output matrix
    # matrix dimensions (input)
    M: tl.constexpr,  # number of elements in M-th dimension
    N: tl.constexpr,  # number of elements in N-th dimension
    K: tl.constexpr,  # number of elements in K-th dimension
    # block sizes (input)
    BLOCK_SIZE_N: tl.constexpr,  # number of elements in block on N-th dimension
    BLOCK_SIZE_K: tl.constexpr,  # number of elements in block on K-th dimension
):
    # input block shape: (1, N, BLOCK_SIZE_K)

    pid = tl.program_id(axis=0)  # i-th block of input

    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_k = pid % tl.cdiv(K, BLOCK_SIZE_K)

    block_start_n = (
        0  # assuming that the entire reduction dimension fits within one thread block
    )
    block_start_k = pid_k * BLOCK_SIZE_K

    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)

    mask_n = offsets_n < N
    mask_k = offsets_k < K

    # idxs has shape (N, BLOCK_SIZE_K)
    idxs_base = (offsets_n[:, None] * K) + offsets_k
    idxs = idxs_base + (
        pid_m * N * K
    )  # increment idxs by the number of elements in all previous blocks

    # mask has shape (N, BLOCK_SIZE_K)
    mask = mask_n[:, None] & mask_k

    # loaded pointers have shape (N, K)
    input = tl.load(
        input_ptr + idxs, mask=mask, other=0
    )  # zero out masked values from input

    # output has shape (1, BLOCK_SIZE_K)
    output = tl.sum(input, axis=0)

    output_offsets = (pid_m * K) + offsets_k

    # stored pointers have shape (1, BLOCK_SIZE_K)
    tl.store(
        output_ptr + output_offsets, output, mask=mask_k
    )  # store a 1D vector into a specific row of 2D output
