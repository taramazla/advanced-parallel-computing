#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// #define N 2048  // Matrix dimension
// #define TILE_SIZE 32  // Block size for CUDA

void count_matmul(double *A, double *B, double *C, int size) {
  for (int row = 0; row < size; row++) {
    for (int col = 0; col < size; col++) {
      if (row < size && col < size) {
        double sum = 0.0;
        for (int k = 0; k < size; ++k) {
          sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
      }
    }
  }
}

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

void transposeMatrix(double *matrix, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      double temp = matrix[i * size + j];
      matrix[i * size + j] = matrix[j * size + i];
      matrix[j * size + i] = temp;
    }
  }
}

int main(int argc, char **argv) {
  // default value for N and tile size
  int N = 128;

  if (argc == 2) {
    N = atoi(argv[1]);
  } else {
    printf("Custom block size? pass [N] as arguments\n\n");
  }
  printf("N = %d;\n", N);

  size_t matrixSize = N * N * sizeof(double);
  double *h_A, *h_B, *h_C; // Host matrices

  // Allocate host memory
  h_A = (double *)malloc(matrixSize);
  h_B = (double *)malloc(matrixSize);
  h_C = (double *)malloc(matrixSize);

  // Initialize matrices
  initializeMatrix(h_A, N, 1.0);
  initializeMatrix(h_B, N, 2.0);
  initializeMatrix(h_C, N, 0.0);

  // Transpose matrix B for better memory access pattern
  transposeMatrix(h_B, N);
  //
  // Start timer
  struct timeval totalStart, totalStop;
  gettimeofday(&totalStart, NULL);

  count_matmul(h_A, h_B, h_C, N);

  // Stop total timer
  gettimeofday(&totalStop, NULL);

  // Calculate total and communication time
  double totalTime = (totalStop.tv_sec + totalStop.tv_usec * 1e-6) -
                     (totalStart.tv_sec + totalStart.tv_usec * 1e-6);

  // Print results
  printf("Matrix Size (N) = %d\n", N);
  printf("\tTotal Execution Time = %.6f seconds\n", totalTime);
  printf("\tComputation Time = %.6f seconds\n", totalTime);

  // Print first 10x10 of result matrix
  printf("\nResult matrix (first 10x10):\n");
  printMatrix(h_C, N, 10);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}