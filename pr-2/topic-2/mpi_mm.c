// mpi_mm.c - Matrix multiplication with MPI (cluster / multicore)
//
// Compile:
//   mpicc mpi_mm.c -O3 -o mpi_mm
//
// Run (contoh):
//   mpirun -np 4 ./mpi_mm 1024
//
// Program ini menghitung C = A x B, dengan A,B,C disimpan row-major.
// Distribusi:
//   - A dibagi per baris ke semua proses (MPI_Scatterv)
//   - B dibroadcast ke semua proses (MPI_Bcast)
//   - C digabung kembali (MPI_Gatherv)
//
// Rank 0 juga melakukan verifikasi (maksimal error) terhadap versi
// sekuensial CPU jika N <= 512.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void count_matmul(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

void printMatrix(double *M, int N) {
    for (int i = 0; i < (N < 10 ? N : 10); i++) {
        for (int j = 0; j < (N < 10 ? N : 10); j++) {
            printf("%.2f ", M[i*N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) printf("Usage: ./mpi_mm N\n");
        MPI_Finalize();
        return 0;
    }

    int N = atoi(argv[1]);
    int total_elems = N * N;

    double *A = NULL;
    double *B = (double *)malloc(total_elems * sizeof(double));
    double *C = NULL;
    double *C_ref = NULL;

    // Local buffers
    int base = N / size;
    int extra = N % size;
    int my_rows = base + (rank < extra ? 1 : 0);
    int my_offset = rank * base + (rank < extra ? rank : extra);

    double *A_local = (double *)malloc(my_rows * N * sizeof(double));
    double *C_local = (double *)malloc(my_rows * N * sizeof(double));

    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        A = (double *)malloc(total_elems * sizeof(double));
        C = (double *)malloc(total_elems * sizeof(double));
        C_ref = (double *)malloc(total_elems * sizeof(double));

        // Init A and B
        for (int i = 0; i < total_elems; i++) {
            A[i] = 1.0;
            B[i] = 2.0;
        }

        sendcounts = malloc(size * sizeof(int));
        displs     = malloc(size * sizeof(int));

        for (int r = 0; r < size; r++) {
            int rows_r = base + (r < extra ? 1 : 0);
            sendcounts[r] = rows_r * N;
        }

        displs[0] = 0;
        for (int r = 1; r < size; r++) {
            displs[r] = displs[r-1] + sendcounts[r-1];
        }
    }

    // 🟩 Broadcast B ke semua rank
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 🟩 Scatterv A ke worker
    MPI_Scatterv(
        A, sendcounts, displs, MPI_DOUBLE,
        A_local, my_rows*N, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // 🟩 Hitung C lokal = A_local × B
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A_local[i*N + k] * B[k*N + j];
            }
            C_local[i*N + j] = sum;
        }
    }

    double t1 = MPI_Wtime();

    // 🟩 Gabungkan hasil dari semua proses
    MPI_Gatherv(
        C_local, my_rows*N, MPI_DOUBLE,
        C, sendcounts, displs, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        printf("MPI Matrix Multiplication N=%d size=%d\n", N, size);
        printf("Compute time: %.6f s\n", t1 - t0);

        printf("\nC (first 10x10):\n");
        printMatrix(C, N);

        // Verifikasi
        count_matmul(A, B, C_ref, N);

        double max_err = 0.0;
        for (int i = 0; i < total_elems; i++) {
            double d = fabs(C[i] - C_ref[i]);
            if (d > max_err) max_err = d;
        }

        printf("\nMax error vs CPU = %e\n", max_err);
    }

    free(A_local);
    free(C_local);
    free(B);

    if (rank == 0) {
        free(A);
        free(C);
        free(C_ref);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
