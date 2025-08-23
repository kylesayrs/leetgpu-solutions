import torch

import triton
import triton.language as tl

@triton.jit
def transpose_2d_grid_kernel(x_ptr, y_ptr, x_height, x_width, BLOCK_SIZE: tl.constexpr):
    x_row_start = tl.program_id(0) * BLOCK_SIZE
    x_col_start = tl.program_id(1) * BLOCK_SIZE

    # always returns (BLOCK_SIZE, BLOCK_SIZE) shape (boundary check is later)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,                 # standard access pattern
        shape=(x_height, x_width),  # for each x0 dim, increment address by x_width (move in direction of x0 dim)
        strides=(x_width, 1),       # for each x1 dim, increment address by 1       (move in direction of x1 dim)
        offsets=(x_row_start, x_col_start),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
        order=(0, 1),  # compiler hint := greatest stride dim first
                       # https://www.mengyibai.com/p/order-in-triton-make-block-ptr/
    )

    y_height: tl.constexpr = x_width
    y_width: tl.constexpr = x_height

    # always returns (BLOCK_SIZE, BLOCK_SIZE) shape (boundary check is later)
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,                 # trick `make_block_ptr`
        shape=(x_height, x_width),  # for each x0 dim, increment address by 1       (move in direction of y1 dim)
        strides=(1, y_width),       # for each x1 dim, increment address by y_width (move in direction of y0 dim)
        offsets=(x_row_start, x_col_start),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
        order=(1, 0),
    )
    
    x_block = tl.load(x_block_ptr, boundary_check=(0, 1))  # when dealing with block ptrs, use a boundary check instead of a mask
    tl.store(y_block_ptr, x_block, boundary_check=(0, 1))  # when dealing with block ptrs, use a boundary check instead of a mask

def transpose_2d_grid(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros((x.size(1), x.size(0)), dtype=x.dtype, device=x.device)
    
    grid = lambda meta: (
        triton.cdiv(x.size(0), meta['BLOCK_SIZE']),
        triton.cdiv(x.size(1), meta['BLOCK_SIZE'])
    )
    transpose_2d_grid_kernel[grid](x, y, x.size(0), x.size(1), BLOCK_SIZE=2)

    return y


if __name__ == "__main__":
    x = torch.arange(15, device="cuda").reshape((3, 5)).contiguous()
    print(x)
    y = transpose_2d_grid(x)
    print(y)