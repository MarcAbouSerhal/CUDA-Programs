
#include "common.h"
#include "timer.h"

#include <cuda/atomic>

#define COARSENING_FACTOR 64

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    __shared__ unsigned int bins_s[NUM_BINS];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        bins_s[i] = 0;
    }
    __syncthreads();

    if(index < width * height) {
        unsigned char b = image[index];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
        bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if(bins_s[i] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[i]);
            bins_ref.fetch_add(bins_s[i], cuda::memory_order_relaxed);
        }
    }
}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    int numThreadsPerBlock = 512;
    int numBlocks = (width * height + numThreadsPerBlock - 1) / numThreadsPerBlock;
    histogram_private_kernel <<< numBlocks, numThreadsPerBlock >>> (image_d, bins_d, width, height);

}

__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    __shared__ unsigned int bins_s[NUM_BINS];


    unsigned int localIndex = threadIdx.x;
    unsigned int blockIndex = blockIdx.x;
    unsigned int blockSize = blockDim.x;
    for (int i = localIndex; i < NUM_BINS; i += blockSize) {
        bins_s[i] = 0;
    }
    __syncthreads();

    for(int segmentStart = blockIndex * blockSize * COARSENING_FACTOR; 
        segmentStart < width * height && segmentStart < (blockIndex + 1) * blockSize * COARSENING_FACTOR; 
        segmentStart += 4 * blockSize
    ) {
        int i = segmentStart + 4 * localIndex;
        if(i + 3 < width * height) {
            char4 image4 = ((char4*)image)[i / 4];
            unsigned char b = image4.x;
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref1(bins_s[b]);
            bins_s_ref1.fetch_add(1, cuda::memory_order_relaxed);
            b = image4.y;
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref2(bins_s[b]);
            bins_s_ref2.fetch_add(1, cuda::memory_order_relaxed);
            b = image4.z;
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref3(bins_s[b]);
            bins_s_ref3.fetch_add(1, cuda::memory_order_relaxed);
            b = image4.w;
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref4(bins_s[b]);
            bins_s_ref4.fetch_add(1, cuda::memory_order_relaxed);
        }
        else {
            for(int j = i; j < width * height; ++j) {
                unsigned char b = image[j];
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
                bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
            }
        }
    }

    
    __syncthreads();
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if(bins_s[i] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[i]);
            bins_ref.fetch_add(bins_s[i], cuda::memory_order_relaxed);
        }
    }
}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    int numThreadsPerBlock = 512;
    int numBlocks = (width * height + numThreadsPerBlock * COARSENING_FACTOR - 1) / (numThreadsPerBlock * COARSENING_FACTOR);
    histogram_private_coarse_kernel <<< numBlocks, numThreadsPerBlock >>> (image_d, bins_d, width, height);

}

