// seq.cu - Matrix multiplication sequential (CPU)
//
// Compile (with nvcc or gcc):
//   nvcc seq.cu -o seq
//   ./seq 1024
//
// Menghitung C = A x B (row-major), A,B diisi konstan.

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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
    int N = 512;  // default
    if (argc >= 2) {
        N = atoi(argv[1]);
    } else {
        printf("Usage: %s [N]\nDefault N = %d\n\n", argv[0], N);
    }

    printf("Sequential CPU matmul, N = %d\n", N);

    size_t bytes = (size_t)N * (size_t)N * sizeof(double);

    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);

    if (!A || !B || !C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Inisialisasi matriks (boleh diubah ke random kalau mau)
    initializeMatrix(A, N, 1.0);
    initializeMatrix(B, N, 2.0);

    struct timeval tStart, tStop;
    gettimeofday(&tStart, NULL);
    count_matmul(A, B, C, N);
    gettimeofday(&tStop, NULL);

    double elapsed = getElapsedMs(tStart, tStop);
    printf("CPU compute time: %.3f ms\n", elapsed);

    printf("\nResult C (first 10x10):\n");
    printMatrix(C, N, 10);

    free(A);
    free(B);
    free(C);

    return 0;
}
