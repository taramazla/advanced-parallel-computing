#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cublas_v2.h>

// #define N 128

void initializeMatrix(double *matrix, int size, double value) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = value;
    }
}

void printMatrix(double *matrix, int size, int printSize) {
    for (int i = 0; i < printSize; i++) {
        for (int j = 0; j < printSize; j++) {
            printf("%.1f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int N = 128;
    if(argc == 2) {
      N = atoi(argv[1]);
    } else {
      printf("Custom N? pass [N] as arguments\n\n");
    }
    printf("N = %d;\n", N);

    double *h_A, *h_B, *h_C;  // Host
    double *d_A, *d_B, *d_C;  // Device

    size_t matrixSize = N * N * sizeof(double);

    // Allocate host memory
    h_A = (double *)malloc(matrixSize);
    h_B = (double *)malloc(matrixSize);
    h_C = (double *)malloc(matrixSize);

    // Initialize matrices
    initializeMatrix(h_A, N, 1.0);
    initializeMatrix(h_B, N, 2.0);
    memset(h_C, 0, matrixSize);

    // Allocate device memory
    cudaMalloc((void **)&d_A, matrixSize);
    cudaMalloc((void **)&d_B, matrixSize);
    cudaMalloc((void **)&d_C, matrixSize);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timer
    struct timeval totalStart, totalStop;
    gettimeofday(&totalStart, NULL);

    // Record computation start
    cudaEventRecord(start);

    // Perform matrix multiplication using cuBLAS
    // C = alpha * A * B + beta * C
    const double alpha = 1.0;
    const double beta = 0.0;
    // cuBLAS is column-major by default
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C, N);

    // Record computation end
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate computation time
    float compTime;
    cudaEventElapsedTime(&compTime, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

    // Stop total timer
    gettimeofday(&totalStop, NULL);

    // Calculate total and communication time
    double totalTime = (totalStop.tv_sec + totalStop.tv_usec * 1e-6) -
                       (totalStart.tv_sec + totalStart.tv_usec * 1e-6);
    double commTime = totalTime - compTime / 1000.0;

    // Print results
    printf("Matrix Size (N) = %d\n", N);
    printf("\tTotal Execution Time = %.6f seconds\n", totalTime);
    printf("\tComputation Time = %.6f seconds\n", compTime / 1000.0);
    printf("\tCommunication Time = %.6f seconds\n", commTime);

    printf("\nResult matrix (first 10x10):\n");
    printMatrix(h_C, N, 10);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}