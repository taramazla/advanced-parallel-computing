#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 256  // number of rows and columns in matrix

double comm_total = 0.0;

int main() {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time, comm_start, comm_end;

    // Dynamically allocate matrices
    int* A = NULL;
    int* B = malloc(N * N * sizeof(int));
    int* C = NULL;

    if (rank == 0) {
        A = malloc(N * N * sizeof(int));
        C = malloc(N * N * sizeof(int));
        for (int i = 0; i < N * N; i++) {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
    }

    // Broadcast B to all processes
    comm_start = MPI_Wtime();
    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    comm_end = MPI_Wtime();
    comm_total += comm_end - comm_start;

    start_time = MPI_Wtime();

    // Scatter rows of A
    int rows_per_process = N / size;
    int* sub_A = malloc(rows_per_process * N * sizeof(int));

    comm_start = MPI_Wtime();
    MPI_Scatter(A, rows_per_process * N, MPI_INT, sub_A, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);
    comm_end = MPI_Wtime();
    comm_total += comm_end - comm_start;

    // Allocate and initialize sub_C
    int* sub_C = malloc(rows_per_process * N * sizeof(int));
    memset(sub_C, 0, rows_per_process * N * sizeof(int));

    // Matrix multiplication
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                sub_C[i * N + j] += sub_A[i * N + k] * B[k * N + j];
            }
        }
    }

    // Gather results
    comm_start = MPI_Wtime();
    MPI_Gather(sub_C, rows_per_process * N, MPI_INT, C, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);
    comm_end = MPI_Wtime();
    comm_total += comm_end - comm_start;

    end_time = MPI_Wtime();

    // Reduce communication time across all processes
    double global_comm_total = 0.0;
    MPI_Reduce(&comm_total, &global_comm_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Execution time result
    if (rank == 0) {
        double total = end_time - start_time;
        double avg_comm_time = global_comm_total / size;

        printf("Matrix Size (N) = %d, Processors = %d\n", N, size);
        printf("Total Time: %.6f\n", total);
        printf("Communication Overhead: %.6f\n", avg_comm_time);
        printf("Computation Time (Max): %.6f\n", total - avg_comm_time);

        double computation_time = total - avg_comm_time;
        double pe = (total > 0) ? (computation_time / total) * 100.0 : 0.0;
        printf("Parallel Efficiency: %.2f%%\n", pe);

        printf("Result Matrix C:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", C[i * N + j]);
            }
            printf("\n");
        }

    }


    // Free memory
    free(B);
    free(sub_A);
    free(sub_C);
    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
}