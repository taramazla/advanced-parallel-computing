// opt.cu - Matrix multiplication CUDA dengan shared memory (tiled)
//
// Compile:
//   nvcc -DTILE_SIZE=16 opt.cu -o opt16
//   nvcc -DTILE_SIZE=32 opt.cu -o opt32
// Run:
//   ./opt16 [N]
//
// Menghitung C = A x B, row-major.
// Kernel aman untuk N yang tidak habis dibagi TILE_SIZE.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

void count_matmul(double *A, double *B, double *C, int size) {
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += A[row * size + k] * B[k * size + col];
            }
            C[row * size + col] = sum;
        }
    }
}

void initializeMatrix(double *matrix, int size, double value) {
    int n2 = size * size;
    for (int i = 0; i < n2; i++) {
        matrix[i] = value;
    }
}

void printMatrix(double *matrix, int size, int printSize) {
    if (printSize > size) printSize = size;
    for (int i = 0; i < printSize; i++) {
        for (int j = 0; j < printSize; j++) {
            printf("%.1f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

double getElapsedMs(struct timeval start, struct timeval stop) {
    return (stop.tv_sec - start.tv_sec) * 1000.0 +
           (stop.tv_usec - start.tv_usec) / 1000.0;
}

// Kernel tiled dengan shared memory
__global__ void matrixMulTiledKernel(double *A, double *B, double *C, int size) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;

    int numTiles = (size + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int Acol = t * TILE_SIZE + threadIdx.x;
        int Brow = t * TILE_SIZE + threadIdx.y;

        if (row < size && Acol < size)
            As[threadIdx.y][threadIdx.x] = A[row * size + Acol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (Brow < size && col < size)
            Bs[threadIdx.y][threadIdx.x] = B[Brow * size + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < size && col < size) {
        C[row * size + col] = sum;
    }
}

int main(int argc, char **argv) {
    int N = 1024;
    if (argc >= 2) {
        N = atoi(argv[1]);
    }

    printf("CUDA matmul (shared memory), N = %d, TILE_SIZE = %d\n", N, TILE_SIZE);

    size_t bytes = (size_t)N * (size_t)N * sizeof(double);

    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);
    double *h_C_ref = (double*)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    initializeMatrix(h_A, N, 1.0);
    initializeMatrix(h_B, N, 2.0);
    memset(h_C, 0, bytes);
    memset(h_C_ref, 0, bytes);

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    if (!d_A || !d_B || !d_C) {
        fprintf(stderr, "Device malloc failed\n");
        return 1;
    }

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    struct timeval totalStart, totalStop;
    gettimeofday(&totalStart, NULL);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMulTiledKernel<<<grid, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float computeMs = 0.0f;
    cudaEventElapsedTime(&computeMs, start, stop);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    gettimeofday(&totalStop, NULL);
    double totalMs = getElapsedMs(totalStart, totalStop);
    double commMs = totalMs - computeMs;

    // Print results
    printf("Nama: Tara Mazaya Lababan\nNPM: 2406514564\n");
    printf("Matrix Size (N) = %d\n", N);
    printf("\tTotal Execution Time = %.3f ms\n", totalMs);
    printf("\tComputation Time = %.3f ms\n", computeMs);
    printf("\tCommunication Time = %.3f ms\n", commMs);

    printf("\nResult matrix (first 10x10):\n");
    printMatrix(h_C, N, 10);

    if (N <= 512) {
        count_matmul(h_A, h_B, h_C_ref, N);
        double max_err = 0.0;
        for (int i = 0; i < N * N; ++i) {
            double diff = h_C[i] - h_C_ref[i];
            if (diff < 0) diff = -diff;
            if (diff > max_err) max_err = diff;
        }
        printf("\nMax abs error vs CPU = %e\n", max_err);
    } else {
        printf("\nSkip CPU verification (N > 512), jalankan seq.cu jika ingin cek.\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
