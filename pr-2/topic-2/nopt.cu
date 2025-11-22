#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// #define N 2048  // Matrix dimension
// #define TILE_SIZE 32  // Block size for CUDA

// CUDA kernel for matrix multiplication (non-shared memory)
__global__ void matrixMulKernel(double *A, double *B, double *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        double sum = 0.0;
        for (int k = 0; k < size; ++k) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

void initializeMatrix(double *matrix, int size, double value) {
    for (int i = 0; i < size*size; i++) {
        matrix[i] = value;
    }
}

void printMatrix(double *matrix, int size, int printSize) {
    for (int i = 0; i < printSize; i++) {
        for (int j = 0; j < printSize; j++) {
            printf("%.1f ", matrix[i*size + j]);
        }
        printf("\n");
    }
}

void transposeMatrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = i+1; j < size; j++) {
            double temp = matrix[i*size + j];
            matrix[i*size + j] = matrix[j*size + i];
            matrix[j*size + i] = temp;
        }
    }
}

int main(int argc, char **argv) {
    // default value for N and tile size
    int N = 128;
    int TILE_SIZE = 32;

    if(argc == 3) {
      N = atoi(argv[1]);
      TILE_SIZE = atoi(argv[2]);
    } else {
      printf("Custom block size? pass [N] [blockSize] as arguments\n\n");
    }
    printf("N = %d; blockSize = %d;\n", N, TILE_SIZE);


    double *h_A, *h_B, *h_C;  // Host matrices
    double *d_A, *d_B, *d_C;  // Device matrices

    size_t matrixSize = N * N * sizeof(double);

    // Allocate host memory
    h_A = (double *)malloc(matrixSize);
    h_B = (double *)malloc(matrixSize);
    h_C = (double *)malloc(matrixSize);

    // Initialize matrices
    initializeMatrix(h_A, N, 1.0);
    initializeMatrix(h_B, N, 2.0);
    memset(h_C, 0, matrixSize);

    // Transpose matrix B for better memory access pattern
    transposeMatrix(h_B, N);

    // Allocate device memory
    cudaMalloc((void **)&d_A, matrixSize);
    cudaMalloc((void **)&d_B, matrixSize);
    cudaMalloc((void **)&d_C, matrixSize);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    // Set up CUDA grid and block dimensions
    dim3 dimGrid((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timer
    struct timeval totalStart, totalStop;
    gettimeofday(&totalStart, NULL);

    // Record CUDA computation start time
    cudaEventRecord(start);

    // Launch kernel
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Record CUDA computation end time
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
    double totalTime = (totalStop.tv_sec + totalStop.tv_usec*1e-6) -
                      (totalStart.tv_sec + totalStart.tv_usec*1e-6);
    double commTime = totalTime - compTime/1000.0;  // Convert ms to s

    // Print results
    printf("Matrix Size (N) = %d\n", N);
    printf("\tTotal Execution Time = %.6f seconds\n", totalTime);
    printf("\tComputation Time = %.6f seconds\n", compTime/1000.0);
    printf("\tCommunication Time = %.6f seconds\n", commTime);

    // Print first 10x10 of result matrix
    printf("\nResult matrix (first 10x10):\n");
    printMatrix(h_C, N, 10);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}