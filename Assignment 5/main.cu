
#include "common.h"
#include "timer.h"

#include <cuda/atomic>

__global__ void histogram_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < width*height) {
        unsigned char b = image[i];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
        bins_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
}

void histogram_gpu(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // Call kernel
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (width*height + numThreadsPerBlock - 1)/numThreadsPerBlock;
    histogram_kernel <<< numBlocks, numThreadsPerBlock >>> (image_d, bins_d, width, height);

}

void histogram_cpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    for(unsigned int i = 0; i < width*height; ++i) {
        unsigned char b = image[i];
        ++bins[b];
    }
}

void verify(unsigned int* bins_cpu, unsigned int* bins_gpu) {
    for (unsigned int b = 0; b < NUM_BINS; ++b) {
        if(bins_cpu[b] != bins_gpu[b]) {
            printf("Mismatch at bin %u (CPU result = %u, GPU result = %u)\n", b, bins_cpu[b], bins_gpu[b]);
            return;
        }
    }
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int height = (argc > 1)?(atoi(argv[1])):4096;
    unsigned int width = (argc > 2)?(atoi(argv[2])):4096;
    unsigned char* image = (unsigned char*) malloc(width*height*sizeof(unsigned char));
    unsigned int* bins_cpu = (unsigned int*) malloc(NUM_BINS*sizeof(unsigned int));
    unsigned int* bins_gpu = (unsigned int*) malloc(NUM_BINS*sizeof(unsigned int));
    for (unsigned int row = 0; row < height; ++row) {
        for (unsigned int col = 0; col < width; ++col) {
            image[row*width + col] = rand()%256;
        }
    }
    memset(bins_cpu, 0, NUM_BINS*sizeof(unsigned int));

    // Compute on CPU
    startTime(&timer);
    histogram_cpu(image, bins_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU memory
    cudaEventRecord(start);
    unsigned char *image_d;
    unsigned int *bins_d;
    cudaMalloc((void**) &image_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &bins_d, NUM_BINS*sizeof(unsigned int));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Allocation time: %.3f ms\n", elapsedTime);

    // Copy data to GPU
    cudaEventRecord(start);
    cudaMemcpy(image_d, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(bins_d, 0, NUM_BINS*sizeof(unsigned int));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy to GPU time: %.3f ms\n", elapsedTime);

    // Compute on GPU
    cudaEventRecord(start);
    histogram_gpu(image_d, bins_d, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[1;32mGPU kernel time (unoptimized): %.3f ms\033[0m\n", elapsedTime);

    // Copy data from GPU
    cudaEventRecord(start);
    cudaMemcpy(bins_gpu, bins_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU time: %.3f ms\n", elapsedTime);

    // Verify result
    verify(bins_cpu, bins_gpu);
    memset(bins_gpu, 0, NUM_BINS*sizeof(unsigned int));

    // Compute on GPU
    cudaMemset(bins_d, 0, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    histogram_gpu_private(image_d, bins_d, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[1;32mGPU kernel time (with privatization and shared memory): %.3f ms\033[0m\n", elapsedTime);

    // Copy data from GPU
    cudaEventRecord(start);
    cudaMemcpy(bins_gpu, bins_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU time: %.3f ms\n", elapsedTime);

    // Verify result
    verify(bins_cpu, bins_gpu);
    memset(bins_gpu, 0, NUM_BINS*sizeof(unsigned int));

    // Compute on GPU
    cudaMemset(bins_d, 0, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    histogram_gpu_private_coarse(image_d, bins_d, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[1;32mGPU kernel time (with privatization, shared memory, and thread coarsening): %.3f ms\033[0m\n", elapsedTime);

    // Copy data from GPU
    cudaEventRecord(start);
    cudaMemcpy(bins_gpu, bins_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU time: %.3f ms\n", elapsedTime);

    // Verify result
    verify(bins_cpu, bins_gpu);

    // Free GPU memory
    cudaEventRecord(start);
    cudaFree(image_d);
    cudaFree(bins_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Deallocation time: %.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free memory
    free(image);
    free(bins_cpu);
    free(bins_gpu);

    return 0;

}

