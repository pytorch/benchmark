import torch
import triton
import triton.language as tl


@triton.jit
def triton_sum_kernel_scalar(
    input_ptr,
    output_ptr,
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
