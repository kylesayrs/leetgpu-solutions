#include <cuda_runtime.h>

inline int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    // load each pixel (RGBA) and subtract 255 from RGB
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id < width * height) {
        u_int32_t *pixels = reinterpret_cast<u_int32_t *>(image);
        u_int32_t pixel = pixels[thread_id];
        pixels[thread_id] = (pixel & 0xFF000000u) | (~pixel & 0x00FFFFFFu);
    }
}


// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = cdiv(width * height, threadsPerBlock);

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
