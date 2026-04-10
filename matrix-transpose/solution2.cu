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


constexpr int M = 3;  // output dim
constexpr int N = 5;  // input dim

template <size_t M, size_t N>
void print2DFloatArray(const float (&array)[M][N], int precision = 2);


void __global__ matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float shared[16][17]; // +1 padding to avoid bank conflicts

    int read_row = blockDim.y * blockIdx.y + threadIdx.y;
    int read_col = blockDim.x * blockIdx.x + threadIdx.x;

    if (read_row < rows && read_col < cols) {
        shared[threadIdx.x][threadIdx.y] = input[read_row * cols + read_col];
    }

    __syncthreads();

    // Swapped block indices for output tile
    int write_row = blockDim.x * blockIdx.x + threadIdx.y;
    int write_col = blockDim.y * blockIdx.y + threadIdx.x;

    if (write_row < cols && write_col < rows) {
        output[write_row * rows + write_col] = shared[threadIdx.y][threadIdx.x];
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);  // threads_per_block = prod
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,  // ceiling division
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);  // ceiling division

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}


int main() {
    float input[M][N] = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
    float output[N][M];
    constexpr int input_size = M * N * sizeof(float);
    constexpr int output_size = N * M * sizeof(float);  // same as input

    float *d_input;
    float *d_output;
    CUDACHECK(cudaMalloc(&d_input, input_size));
    CUDACHECK(cudaMalloc(&d_output, output_size));

    CUDACHECK(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));

    solve(d_input, d_output, M, N);

    CUDACHECK(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));

    print2DFloatArray(input);
    print2DFloatArray(output);

    CUDACHECK(cudaFree(d_input));
    CUDACHECK(cudaFree(d_output));
    return 0;
}


template <size_t M, size_t N>
void print2DFloatArray(const float (&array)[M][N], int precision) {
    char format[10];
    std::snprintf(format, sizeof(format), "%%.%df ", precision);

    for (int i = 0; i < M; ++i) {
        printf("[ ");
        for (int j = 0; j < N; ++j) {
            printf(format, array[i][j]);
        }
        printf("]\n");
    }
}