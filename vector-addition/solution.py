import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # int32 Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    # determine which elements to process
    pid = tl.program_id(axis=0)  # 1d grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # load with mask
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # perform operation
    output = x + y

    # store result
    tl.store(output_ptr + offsets, output, mask=mask)


def add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # allocate output memory on device
    C = torch.empty_like(A)
    N = C.numel()

    # determine launch grid and launch
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    add_kernel[grid](A, B, C, N, BLOCK_SIZE=1024)

    # return result
    return C


if __name__ == "__main__":
    A = torch.tensor([1, 2, 3, 4], device="cuda")
    B = torch.tensor([2, 2, 2, 2], device="cuda")
    C = add(A, B)
    print(C)