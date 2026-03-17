
#include "common.h"
#include "timer.h"

__global__ void scan_kernel(float* input, float* output, unsigned int N) {

    __shared__ float buffer[BLOCK_DIM * 2];

    int threadIndex = threadIdx.x;
    int segmentStart = 2 * BLOCK_DIM * blockIdx.x;

    // Copy onto buffer[2 * threadIndex] and buffer[2 * threadIndex + 1]
    if(segmentStart + 2 * threadIndex < N) {
        buffer[2 * threadIndex] = input[segmentStart + 2 * threadIndex];
    }
    else {
        buffer[2 * threadIndex] = 0.0f;
    }
    if(segmentStart + 2 * threadIndex + 1 < N) {
        buffer[2 * threadIndex + 1] = input[segmentStart + 2 * threadIndex + 1];
    }
    else {
        buffer[2 * threadIndex + 1] = 0.0f;
    }
    __syncthreads();

    // (i, s) -> 2 * (i + 1) * s - 1 (-s)

    // Reduction tree
    for(int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
        if(threadIndex < BLOCK_DIM / stride) {
            buffer[2 * (threadIndex + 1) * stride - 1] += buffer[2 * (threadIndex + 1) * stride - stride - 1];
        }
        __syncthreads();
    }

    // Setting last value to 0
    if(threadIndex == 0) {
        buffer[BLOCK_DIM * 2 - 1] = 0.0f;
    }
    __syncthreads();

    // Post-reduction stage
    for(int stride = BLOCK_DIM; stride > 0; stride /= 2) {
        if(threadIndex < BLOCK_DIM / stride) {
            int rightIndex = 2 * (threadIndex + 1) * stride - 1;
            int leftIndex = rightIndex - stride;
            float oldRightValue = buffer[rightIndex];
            buffer[rightIndex] += buffer[leftIndex];
            buffer[leftIndex] = oldRightValue;
        }
        __syncthreads();
    }

    // Copy from buffer[2 * threadIndex] and buffer[2 * threadIndex + 1]
    if(segmentStart + 2 * threadIndex < N) {
        output[segmentStart + 2 * threadIndex] = buffer[2 * threadIndex];
    }
    if(segmentStart + 2 * threadIndex + 1 < N) {
        output[segmentStart + 2 * threadIndex + 1] = buffer[2 * threadIndex + 1];
    }
}

void scan_gpu_d(float* input_d, float* output_d, unsigned int N) {

    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2*numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;
    scan_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, N);

}

