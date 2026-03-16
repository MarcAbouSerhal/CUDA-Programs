
#include "common.h"
#include "timer.h"

void scan_cpu(float* input, float* output, unsigned int N) {
    for(unsigned int i = 0; i < N; ++i) {
        if(i%(2*BLOCK_DIM) == 0) {
            output[i] = 0.0f;
        } else {
            output[i] = output[i - 1] + input[i - 1];
        }
    }
}

void scan_gpu(float* input, float* output, unsigned int N) {

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory
    cudaEventRecord(start);
    float *input_d, *output_d;
    cudaMalloc((void**) &input_d, N*sizeof(float));
    cudaMalloc((void**) &output_d, N*sizeof(float));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Allocation time: %.3f ms\n", elapsedTime);

    // Copy data to GPU
    cudaEventRecord(start);
    cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy to GPU time: %.3f ms\n", elapsedTime);

    // Compute on GPU
    cudaEventRecord(start);
    scan_gpu_d(input_d, output_d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[1;32mKernel time: %.3f ms\033[0m\n", elapsedTime);

    // Copy data from GPU
    cudaEventRecord(start);
    cudaMemcpy(output, output_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU time: %.3f ms\n", elapsedTime);

    // Free memory
    cudaEventRecord(start);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Deallocation time: %.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int N = (argc > 1)?(atoi(argv[1])):(1 << 24);
    float* input = (float*) malloc(N*sizeof(float));
    float* output_cpu = (float*) malloc(N*sizeof(float));
    float* output_gpu = (float*) malloc(N*sizeof(float));
    for(unsigned int i = 0; i < N; ++i) {
        input[i] = 1.0*rand()/RAND_MAX;
    }

    // Compute on CPU
    startTime(&timer);
    scan_cpu(input, output_cpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Compute on GPU
    startTime(&timer);
    scan_gpu(input, output_gpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Verify result
    for(unsigned int i = 0; i < N; ++i) {
        float diff = (output_cpu[i] - output_gpu[i])/output_cpu[i];
        const float tolerance = 0.0001;
        if(diff > tolerance || diff < -tolerance) {
            printf("Mismatch detected at index %u (CPU result = %e, GPU result = %e)\n", i, output_cpu[i], output_gpu[i]);
            exit(0);
        }
    }

    // Free memory
    free(input);

    return 0;

}

