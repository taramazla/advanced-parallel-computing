// cublas.cu - Matrix multiplication with cuBLAS
//
// Compile:
//   nvcc cublas.cu -lcublas -o cublas
// Run:
//   ./cublas [N]
//
// Menghitung C = A x B, di mana A,B,C disimpan row-major di host.
// Mapping ke cuBLAS (column-major) pakai trik transpose.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

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

int main(int argc, char **argv) {
    int N = 1024;
    if (argc >= 2) {
        N = atoi(argv[1]);
    }

    printf("cuBLAS matmul, N = %d\n", N);

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

    cublasHandle_t handle;
    cublasCreate(&handle);

    struct timeval totalStart, totalStop;
    gettimeofday(&totalStart, NULL);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    double alpha = 1.0;
    double beta  = 0.0;

    // h_A, h_B row-major (N x N).
    // Interpreted as column-major, itu berarti A_col = A_row^T, B_col = B_row^T.
    // Kita ingin C_row = A_row * B_row.
    // Di column-major: C_col = B_col^T * A_col^T sehingga C_col^T = A_row * B_row.
    //
    // Jadi:
    //   C_col = op(A) * op(B) = B^T * A^T
    // di mana A_arg = d_B, B_arg = d_A, dan keduanya di-TRANSPOSE.

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasStatus_t stat = cublasDgemm(handle,
                                      CUBLAS_OP_T, CUBLAS_OP_T,
                                      N, N, N,
                                      &alpha,
                                      d_B, N,   // A (col-major) = B_row^T
                                      d_A, N,   // B (col-major) = A_row^T
                                      &beta,
                                      d_C, N);  // C (col-major) -> C_row

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float computeMs = 0.0f;
    cudaEventElapsedTime(&computeMs, start, stop);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasDgemm failed\n");
        return 1;
    }

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    gettimeofday(&totalStop, NULL);
    double totalMs = getElapsedMs(totalStart, totalStop);
    double commMs = totalMs - computeMs;

    // Print results
    printf("Nama: Tara Mazaya Lababanh\nNPM: 2406514564\n");
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

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
