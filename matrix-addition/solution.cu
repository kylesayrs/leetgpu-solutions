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


inline int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

__global__ void matrix_add(const float* A, const float* B, float* C, int size) {
    // blockDim = threadsPerBlock = 256
    // blockIdx in (0, blocksPerGrid - 1)
    // threadIdx in (0, threadsPerBlock - 1)
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id < size) {
        C[thread_id] = A[thread_id] + B[thread_id];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int size = N * N;
    int blocksPerGrid = cdiv(size, threadsPerBlock);

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, size);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
}

int main() {
    int N = 2;
    float A[N][N] = {{1, 2}, {3, 4}};
    float B[N][N] = {{5, 6}, {7, 8}};
    float C[N][N];

    float *d_A, *d_B, *d_C;
    CUDACHECK(cudaMalloc(&d_A, N * N * sizeof(float)));  // modifies the ptr location of d_A
    CUDACHECK(cudaMalloc(&d_B, N * N * sizeof(float)));  // hence why we need to pass the
    CUDACHECK(cudaMalloc(&d_C, N * N * sizeof(float)));  // pointer itself by reference

    CUDACHECK(cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_A, d_B, d_C, N);

    CUDACHECK(cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f, ", C[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    CUDACHECK(cudaFree(d_A));
    CUDACHECK(cudaFree(d_B));
    CUDACHECK(cudaFree(d_C));
    return 0;
}