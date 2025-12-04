#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Generate a symmetric positive-definite matrix A and vector b
void generate_spd_matrix(double *A, double *b, int N) {
    srand(42);

    // Generate random matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (double)rand() / RAND_MAX;
        }
    }

    // Make it symmetric: A = (A + A^T) / 2
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double avg = (A[i * N + j] + A[j * N + i]) / 2.0;
            A[i * N + j] = avg;
            A[j * N + i] = avg;
        }
    }

    // Make it diagonally dominant
    for (int i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) row_sum += fabs(A[i * N + j]);
        }
        A[i * N + i] = row_sum + 1.0;
    }

    // Generate random RHS
    for (int i = 0; i < N; i++) {
        b[i] = (double)rand() / RAND_MAX;
    }
}

// Local matrix-vector multiplication: y_local = A_local * x_global
void matrix_vector_multiply_local(const double *A_local, const double *x, double *y_local, int local_rows, int N) {
    for (int i = 0; i < local_rows; i++) {
        y_local[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y_local[i] += A_local[i * N + j] * x[j];
        }
    }
}

// Local dot product
double dot_product_local(const double *x_local, const double *y_local, int local_size) {
    double sum = 0.0;
    for (int i = 0; i < local_size; i++) {
        sum += x_local[i] * y_local[i];
    }
    return sum;
}

// Vector AXPY: z = x + alpha*y (local operation)
void vector_axpy_local(double *z_local, const double *x_local, double alpha, const double *y_local, int local_size) {
    for (int i = 0; i < local_size; i++) {
        z_local[i] = x_local[i] + alpha * y_local[i];
    }
}

// Vector scaling: y = alpha*x (local operation)
void vector_scale_local(double *y_local, double alpha, const double *x_local, int local_size) {
    for (int i = 0; i < local_size; i++) {
        y_local[i] = alpha * x_local[i];
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Problem size
    int N = 1024;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // Compute local row distribution
    int local_rows = N / size;
    int remainder = N % size;
    int row_start = rank * local_rows + (rank < remainder ? rank : remainder);
    if (rank < remainder) local_rows++;

    if (rank == 0) {
        printf("MPI Parallel Conjugate Gradient\n");
        printf("Processes: %d\n", size);
        printf("Matrix size: %d x %d\n", N, N);
    }

    // Allocate local memory
    double *A_local = (double*)malloc(local_rows * N * sizeof(double));
    double *b_local = (double*)malloc(local_rows * sizeof(double));
    double *x_local = (double*)malloc(local_rows * sizeof(double));
    double *r_local = (double*)malloc(local_rows * sizeof(double));
    double *p_local = (double*)malloc(local_rows * sizeof(double));
    double *Ap_local = (double*)malloc(local_rows * sizeof(double));

    // Global vectors (needed for matvec)
    double *x_global = (double*)malloc(N * sizeof(double));
    double *p_global = (double*)malloc(N * sizeof(double));

    // Root generates full matrix and distributes
    double *A_full = NULL;
    double *b_full = NULL;
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));

    // Compute sendcounts and displacements for Scatterv
    for (int i = 0; i < size; i++) {
        int rows = N / size;
        if (i < remainder) rows++;
        sendcounts[i] = rows;
        displs[i] = (i < remainder) ? i * rows : i * rows + remainder;
    }

    if (rank == 0) {
        A_full = (double*)malloc(N * N * sizeof(double));
        b_full = (double*)malloc(N * sizeof(double));
        generate_spd_matrix(A_full, b_full, N);
    }

    // Scatter b
    MPI_Scatterv(rank == 0 ? b_full : NULL, sendcounts, displs, MPI_DOUBLE,
                 b_local, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter rows of A
    int *sendcounts_A = (int*)malloc(size * sizeof(int));
    int *displs_A = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        sendcounts_A[i] = sendcounts[i] * N;
        displs_A[i] = displs[i] * N;
    }

    MPI_Scatterv(rank == 0 ? A_full : NULL, sendcounts_A, displs_A, MPI_DOUBLE,
                 A_local, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initialize x = 0
    for (int i = 0; i < local_rows; i++) {
        x_local[i] = 0.0;
    }
    for (int i = 0; i < N; i++) {
        x_global[i] = 0.0;
    }

    // CG parameters
    const double tol = 1e-6;
    const int max_iter = 10000;
    const double EPS_BREAK = 1e-30;

    double r_dot_r, r_dot_r_new, p_dot_Ap, alpha, beta;
    int iter;

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Initialize: r0 = b - A*x0, but since x0 = 0, r0 = b
    for (int i = 0; i < local_rows; i++) {
        r_local[i] = b_local[i];
        p_local[i] = r_local[i]; // p0 = r0
    }

    // Allgather p to get p_global
    MPI_Allgatherv(p_local, local_rows, MPI_DOUBLE,
                   p_global, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    // Initial r_dot_r = r0^T * r0 (global reduction)
    double r_dot_r_local = dot_product_local(r_local, r_local, local_rows);
    MPI_Allreduce(&r_dot_r_local, &r_dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double initial_residual = sqrt(r_dot_r);
    if (rank == 0) {
        printf("Initial residual: %e\n", initial_residual);
    }

    // Main CG loop
    for (iter = 0; iter < max_iter; iter++) {
        // Compute Ap_local = A_local * p_global
        matrix_vector_multiply_local(A_local, p_global, Ap_local, local_rows, N);

        // Compute p_dot_Ap = p^T * Ap (global reduction)
        double p_dot_Ap_local = dot_product_local(p_local, Ap_local, local_rows);
        MPI_Allreduce(&p_dot_Ap_local, &p_dot_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Check for breakdown
        if (fabs(p_dot_Ap) < EPS_BREAK) {
            if (rank == 0) {
                fprintf(stderr, "CG breakdown: p^T A p ~ 0 (pAp=%e) at iter %d\n", p_dot_Ap, iter);
            }
            break;
        }

        // Compute alpha = r_dot_r / p_dot_Ap
        alpha = r_dot_r / p_dot_Ap;

        // Update x_local = x_local + alpha*p_local
        vector_axpy_local(x_local, x_local, alpha, p_local, local_rows);

        // Update r_local = r_local - alpha*Ap_local
        vector_axpy_local(r_local, r_local, -alpha, Ap_local, local_rows);

        // Compute new r_dot_r = r^T * r (global reduction)
        r_dot_r_local = dot_product_local(r_local, r_local, local_rows);
        MPI_Allreduce(&r_dot_r_local, &r_dot_r_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Check convergence
        if (sqrt(r_dot_r_new) < tol * initial_residual) {
            if (rank == 0) {
                printf("Converged after %d iterations\n", iter + 1);
            }
            ++iter;
            r_dot_r = r_dot_r_new;
            break;
        }

        // Compute beta = r_dot_r_new / r_dot_r
        beta = r_dot_r_new / r_dot_r;

        // Update p_local = r_local + beta*p_local
        vector_axpy_local(p_local, r_local, beta, p_local, local_rows);

        // Allgather p to get p_global for next iteration
        MPI_Allgatherv(p_local, local_rows, MPI_DOUBLE,
                       p_global, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        // Update r_dot_r for next iteration
        r_dot_r = r_dot_r_new;
    }

    // Gather x for verification on root
    MPI_Gatherv(x_local, local_rows, MPI_DOUBLE,
                rank == 0 ? x_global : NULL, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // Verify solution on root
    if (rank == 0) {
        double *Ax = (double*)malloc(N * sizeof(double));
        // Simple verification: compute Ax
        for (int i = 0; i < N; i++) {
            Ax[i] = 0.0;
            for (int j = 0; j < N; j++) {
                Ax[i] += A_full[i * N + j] * x_global[j];
            }
        }

        double residual = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = b_full[i] - Ax[i];
            residual += diff * diff;
        }
        residual = sqrt(residual);

        printf("Iterations: %d\n", iter);
        printf("Final residual: %e\n", sqrt(r_dot_r));
        printf("Verification residual: %e\n", residual);
        printf("Total time: %.6f seconds\n", elapsed);

        free(Ax);
        free(A_full);
        free(b_full);
    }

    // Clean up
    free(A_local);
    free(b_local);
    free(x_local);
    free(r_local);
    free(p_local);
    free(Ap_local);
    free(x_global);
    free(p_global);
    free(sendcounts);
    free(displs);
    free(sendcounts_A);
    free(displs_A);

    MPI_Finalize();
    return 0;
}
