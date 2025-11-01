#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define N 256  // global matrix is N x N

static inline int is_perfect_square(int p) {
    int r = (int)(sqrt((double)p) + 0.5);
    return r * r == p ? r : -1;
}

static inline void dgemm_block(int bs, const double *A, const double *B, double *C) {
    // C += A * B, all bs x bs, row-major
    for (int i = 0; i < bs; i++) {
        for (int k = 0; k < bs; k++) {
            double aik = A[i*bs + k];
            for (int j = 0; j < bs; j++) {
                C[i*bs + j] += aik * B[k*bs + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Check process grid ---
    int q = is_perfect_square(size);
    if (q < 0) {
        if (rank == 0) fprintf(stderr, "Error: number of processes (%d) must be a perfect square.\n", size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (N % q != 0) {
        if (rank == 0) fprintf(stderr, "Error: N=%d must be divisible by sqrt(P)=%d.\n", N, q);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int bs = N / q;            // block size per rank (bs x bs)

    // --- Create 2D Cartesian grid with wrap-around (torus) ---
    int dims[2]    = { q, q };
    int periods[2] = { 1, 1 };
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid);

    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    // --- Create row and column sub-communicators (for Fox broadcasts/shifts) ---
    MPI_Comm row_comm, col_comm;
    int remain_dims_row[2] = { 0, 1 }; // keep columns: communicator per row
    int remain_dims_col[2] = { 1, 0 }; // keep rows: communicator per column
    MPI_Cart_sub(grid, remain_dims_row, &row_comm);
    MPI_Cart_sub(grid, remain_dims_col, &col_comm);

    // --- Allocate local blocks ---
    double *Ablock = (double*)calloc((size_t)bs * bs, sizeof(double));
    double *Bblock = (double*)calloc((size_t)bs * bs, sizeof(double));
    double *Cblock = (double*)calloc((size_t)bs * bs, sizeof(double));
    if (!Ablock || !Bblock || !Cblock) {
        fprintf(stderr, "Rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    // Temporary buffer for broadcasted A (Fox)
    double *Atemp = (double*)malloc((size_t)bs * bs * sizeof(double));
    if (!Atemp) {
        fprintf(stderr, "Rank %d: allocation failed (Atemp)\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    // --- Root allocates and initializes full matrices ---
    double *A = NULL, *B = NULL, *C = NULL;
    if (rank == 0) {
        A = (double*)malloc((size_t)N * N * sizeof(double));
        B = (double*)malloc((size_t)N * N * sizeof(double));
        C = (double*)calloc((size_t)N * N, sizeof(double));
        if (!A || !B || !C) {
            fprintf(stderr, "Root: allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i*N + j] = 1.0;
                B[i*N + j] = 2.0;
            }
        }
    }

    // --- Create a block datatype to Scatterv/Gatherv submatrices ---
    MPI_Datatype block_t, block_t_resized;
    MPI_Type_vector(bs, bs, N, MPI_DOUBLE, &block_t);            // rows=bs, cols=bs, stride=N
    MPI_Type_create_resized(block_t, 0, sizeof(double), &block_t_resized);
    MPI_Type_commit(&block_t_resized);

    int *counts = NULL, *displs = NULL;
    if (rank == 0) {
        counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for (int r = 0; r < size; r++) {
            int rc[2] = { r / q, r % q };                         // (row, col) in grid
            counts[r] = 1;
            displs[r] = rc[0] * bs * N + rc[1] * bs;              // top-left of that block in row-major
        }
    }

    // --- Synchronize & start timers ---
    MPI_Barrier(MPI_COMM_WORLD);
    double t0_total = MPI_Wtime();

    // --- Distribute A and B blocks to all ranks ---
    double t0_comm = MPI_Wtime();
    MPI_Scatterv(A, counts, displs, block_t_resized,
                 Ablock, bs*bs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, counts, displs, block_t_resized,
                 Bblock, bs*bs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_comm = MPI_Wtime() - t0_comm;    // will accumulate more comm below

    // =========================
    // Fox Algorithm main loop
    // =========================
    double comp_time_local = 0.0;
    MPI_Status st;

    for (int s = 0; s < q; s++) {
        // Root in this row that broadcasts A for stage s:
        // process in column (row + s) mod q
        int root_col = (my_row + s) % q;

        // Prepare Atemp on the root of the row broadcast
        if (my_col == root_col) {
            // copy local Ablock into Atemp to use as bcast buffer
            memcpy(Atemp, Ablock, (size_t)bs * bs * sizeof(double));
        }

        // Broadcast Atemp across the row
        double t0_bcast = MPI_Wtime();
        MPI_Bcast(Atemp, bs*bs, MPI_DOUBLE, /*root=*/root_col, row_comm);
        t_comm += MPI_Wtime() - t0_bcast;

        // Compute with the broadcasted A and local B
        double t0_comp = MPI_Wtime();
        dgemm_block(bs, Atemp, Bblock, Cblock);
        comp_time_local += (MPI_Wtime() - t0_comp);

        // Shift B up by one (circular) within the column (using the original grid or col_comm)
        int src, dst;
        double t0_shift = MPI_Wtime();
        // Use Cartesian neighbor info for clarity (equivalent to Sendrecv on col_comm ranks ±1)
        MPI_Cart_shift(grid, /*dim=*/0, /*disp=*/-1, &src, &dst); // up by one
        MPI_Sendrecv_replace(Bblock, bs*bs, MPI_DOUBLE, dst, 400,
                             src, 400, grid, &st);
        t_comm += MPI_Wtime() - t0_shift;
    }

    // --- Gather C blocks back to root ---
    double t0_gath = MPI_Wtime();
    MPI_Gatherv(Cblock, bs*bs, MPI_DOUBLE,
                C, counts, displs, block_t_resized,
                0, MPI_COMM_WORLD);
    t_comm += MPI_Wtime() - t0_gath;

    // --- Compute global timing stats ---
    double max_comp, min_comp, sum_comp;
    MPI_Reduce(&comp_time_local, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time_local, &min_comp, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time_local, &sum_comp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_comm;
    MPI_Reduce(&t_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double total_time = MPI_Wtime() - t0_total;

    if (rank == 0) {
        double avg_comp = sum_comp / size;
        double load_imbalance = max_comp - min_comp;
        double comm_overhead = max_comm; // worst-case comm cost among ranks

        printf("Matrix Size (N) = %d, Processors = %d (%d x %d grid)\n", N, size, q, q);
        printf("Tara Mazaya Lababan - 2406514564\n");
        printf("\tTotal Time: %.6f s\n", total_time);
        printf("\tComputation Time (Max): %.6f s\n", max_comp);
        printf("\tComputation Time (Min): %.6f s\n", min_comp);
        printf("\tComputation Time (Avg): %.6f s\n", avg_comp);
        printf("\tCommunication Time (Max across ranks): %.6f s\n", comm_overhead);
        printf("\tLoad Imbalance: %.6f s (%.2f%%)\n",
               load_imbalance,
               max_comp > 0.0 ? (load_imbalance / max_comp * 100.0) : 0.0);
        printf("\tParallel Efficiency: %.2f%%\n",
               total_time > 0.0 ? (max_comp / total_time * 100.0) : 0.0);
        printf("\tCommunication Overhead: %.6f s (%.2f%%)\n\n",
               comm_overhead,
               total_time > 0.0 ? (comm_overhead / total_time * 100.0) : 0.0);

        printf("Result Matrix C:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.1f ", C[i*N + j]);
            }
            printf("\n");
        }
    }

    // --- Cleanup ---
    MPI_Type_free(&block_t_resized);
    if (rank == 0) {
        free(A); free(B); free(C);
        free(counts); free(displs);
    }
    free(Ablock); free(Bblock); free(Cblock);
    free(Atemp);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}
