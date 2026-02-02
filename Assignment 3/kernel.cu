
#include "common.h"
#include "timer.h"

#define TILE_DIM 32

// M x K x N

__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    __shared__ float A_tile[TILE_DIM][TILE_DIM];
    __shared__ float B_tile[TILE_DIM][TILE_DIM];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int relative_row = row % TILE_DIM;
    unsigned int relative_col = col % TILE_DIM;
    float sum = 0;
    for(int i = 0; i < (K + TILE_DIM - 1) / TILE_DIM; ++i) {
        // copy onto SM
        // copy from A_{row, i * TILE_DIM + relative_col}
        if(row < M && i * TILE_DIM + relative_col < K) A_tile[relative_row][relative_col] = A[K * row + i * TILE_DIM + relative_col];
        // copy from B_{i * TILE_DIM + relative_row, col}
        if(i * TILE_DIM + relative_row < K && col < N) B_tile[relative_row][relative_col] = B[N * (i * TILE_DIM + relative_row) + col];
        __syncthreads();
        // add to C_{row, col}
        if(row < M && col < N)
            for(int j = 0; j < min(TILE_DIM, K - i * TILE_DIM); ++j)
                sum += A_tile[relative_row][j] * B_tile[j][relative_col];
        __syncthreads();
    }
    if(row < M && col < N) C[N * row + col] = sum;
}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU memory
    cudaEventRecord(start);

    float* A_d;
    float* B_d;
    float* C_d;
	cudaMalloc((void**)&A_d, M * K * sizeof(float));
	cudaMalloc((void**)&B_d, K * N * sizeof(float));
	cudaMalloc((void**)&C_d, M * N * sizeof(float));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Allocation time: %.3f ms\n", elapsedTime);

    // Copy data to GPU
    cudaEventRecord(start);

    cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy to GPU time: %.3f ms\n", elapsedTime);

    // Call kernel
    cudaEventRecord(start);

    dim3 numThreadsPerBlock(TILE_DIM, TILE_DIM); 
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    mm_tiled_kernel <<< numBlocks, numThreadsPerBlock >>> (A_d, B_d, C_d, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[1;32mKernel time: %.3f ms\033[0m\n", elapsedTime);

    // Copy data from GPU
    cudaEventRecord(start);

    cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU time: %.3f ms\n", elapsedTime);

    // Free GPU memory
    cudaEventRecord(start);

    cudaFree(A_d);
	cudaFree(B_d);
    cudaFree(C_d);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Deallocation time: %.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

