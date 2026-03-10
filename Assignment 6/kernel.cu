
#include "common.h"
#include "timer.h"

#include <cuda/atomic>

#define BLOCK_DIM 1024
#define WARP_SIZE 32

__device__ float warpShuffle(int laneIndex, float val) {
    for(unsigned int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
        float otherVal = __shfl_up_sync(0xffffffff, val, stride);
        if(laneIndex >= stride) {
            val += otherVal;
        }
    }
    return val;
}

__global__ void reduce_kernel(float* input, float* sum, unsigned int N) {

    unsigned int localIndex = threadIdx.x;
    unsigned int globalIndex = 2 * blockIdx.x * BLOCK_DIM + BLOCK_DIM + localIndex;
    unsigned int warpIndex = localIndex / WARP_SIZE;
    unsigned int laneIndex = localIndex % 32;
    
    __shared__ float warpSum[BLOCK_DIM / WARP_SIZE];
    float val;
    if(globalIndex < N){
        val = input[globalIndex] + input[globalIndex - BLOCK_DIM];
    }
    else if(globalIndex - BLOCK_DIM < N) {
        val = input[globalIndex - BLOCK_DIM];
    }
    else {
        val = 0.0f;
    }
    val = warpShuffle(laneIndex, val);
    if(laneIndex == WARP_SIZE - 1) {
        warpSum[warpIndex] = val;
    }
    __syncthreads();
    if(warpIndex == BLOCK_DIM / WARP_SIZE - 1) {
        val = warpSum[laneIndex];
        val = warpShuffle(laneIndex, val);
        if(localIndex == BLOCK_DIM - 1) {
            cuda::atomic_ref<float, cuda::thread_scope_device> sum_ref(*sum);
            sum_ref.fetch_add(val, cuda::memory_order_relaxed);
        }
    }
}

float reduce_gpu(float* input, unsigned int N) {

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory
    cudaEventRecord(start);
    float *input_d;
    cudaMalloc((void**) &input_d, N*sizeof(float));
    float *sum_d;
    cudaMalloc((void**) &sum_d, sizeof(float));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Allocation time: %.3f ms\n", elapsedTime);

    // Copy data to GPU
    cudaEventRecord(start);
    cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(sum_d, 0, sizeof(float));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy to GPU time: %.3f ms\n", elapsedTime);

    // Call kernel
    cudaEventRecord(start);
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2*numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;
    reduce_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, sum_d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[1;32mKernel time: %.3f ms\033[0m\n", elapsedTime);

    // Copy data from GPU
    cudaEventRecord(start);
    float sum;
    cudaMemcpy(&sum, sum_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU time: %.3f ms\n", elapsedTime);

    // Free memory
    cudaEventRecord(start);
    cudaFree(input_d);
    cudaFree(sum_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Deallocation time: %.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return sum;

}

