#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int left = blockIdx.x * blockDim.x + threadIdx.x;
    if (left >= (N >> 1)) return;

    int right = N - 1 - left;

    float tmp = input[left];
    input[left] = input[right];
    input[right] = tmp;
}

extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
