
#include "common.h"
#include "timer.h"

__global__ void mm_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N) {
        int index = row * N + col;
        C[index] = 0;
        for(int k = 0; k < K; ++k)
            C[index] += A[row * K + k] * B[k * N + col];
    }

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

    dim3 numThreadsPerBlock(32, 16); 
    dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (M + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    mm_kernel <<< numBlocks, numThreadsPerBlock >>> (A_d, B_d, C_d, M, N, K);

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

