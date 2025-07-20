#include <iostream>
#include <cuda_runtime.h>

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)


__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;  // 1d grid
    if (thread_id < N) {
        C[thread_id] = A[thread_id] + B[thread_id];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // ceiling divsion

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}


int main() {
    int N = 4;
    float A[] = {1, 2, 3, 4};
    float B[] = {2, 2, 2, 2};
    float C[N];

    float *d_A, *d_B, *d_C;
    CUDACHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_C, N * sizeof(float)));

    CUDACHECK(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_A, d_B, d_C, N);

    CUDACHECK(cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        printf("%f, ", C[i]);
    }
    printf("\n");

    CUDACHECK(cudaFree(d_A));
    CUDACHECK(cudaFree(d_B));
    CUDACHECK(cudaFree(d_C));
    return 0;
}