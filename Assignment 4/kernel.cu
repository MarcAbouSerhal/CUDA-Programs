
#include "common.h"
#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {
    
    __shared__ float input_tile[IN_TILE_DIM][IN_TILE_DIM];

    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;
    int relativeRow = threadIdx.x;
    int relativeCol = threadIdx.y;
    int row = blockRow * OUT_TILE_DIM + relativeRow - FILTER_RADIUS;
    int col = blockCol * OUT_TILE_DIM + relativeCol - FILTER_RADIUS;
    bool inside = row >= 0 && row < height && col >= 0 && col < width;
    if(inside)
        input_tile[relativeRow][relativeCol] = input[row * width + col];
    else 
        input_tile[relativeRow][relativeCol] = 0;
    __syncthreads();
    // middle OUT_TILE_DIM x OUT_TILE_DIM square does computations
    if(relativeRow >= FILTER_RADIUS && relativeRow < IN_TILE_DIM - FILTER_RADIUS && && relativeCol >= FILTER_RADIUS && relativeCol < IN_TILE_DIM - FILTER_RADIUS && inside) {
        float sum = 0.0;
        for(int i = 0; i < FILTER_DIM; ++i)
            for(int j = 0; j < FILTER_DIM; ++j)
                sum += filter_c[i][j] * input_tile[relativeRow + i - FILTER_RADIUS][relativeCol + j - FILTER_RADIUS];
        output[row * width + col] = sum;
    }
}

void copyFilterToGPU(float filter[][FILTER_DIM]) {
    cudaMemcpyToSymbol(
        filter_c,
        filter,
        FILTER_DIM * FILTER_DIM * sizeof(float)
    )
}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {
    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 numBlocks((height + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (width + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, width, height);
}
