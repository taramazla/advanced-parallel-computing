#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

// Get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Generate a symmetric positive-definite matrix A and vector b
void generate_spd_matrix(double *A, double *b, int N) {
    // Initialize random seed
    srand(42);

    // Generate random matrix
    #pragma omp parallel for
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

    // Make it diagonally dominant to ensure positive definiteness
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) row_sum += fabs(A[i * N + j]);
        }
        A[i * N + i] = row_sum + 1.0; // ensures diagonal dominance
    }

    // Generate random right-hand side vector b
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        b[i] = (double)rand() / RAND_MAX;
    }
}

// Matrix-vector multiplication: y = A*x
void matrix_vector_multiply(const double *A, const double *x, double *y, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        y[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y[i] += A[i * N + j] * x[j];
        }
    }
}

// Dot product: result = x^T * y
double dot_product(const double *x, const double *y, int N) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

// Vector addition: z = x + alpha*y
void vector_axpy(double *z, const double *x, double alpha, const double *y, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        z[i] = x[i] + alpha * y[i];
    }
}

// Vector scaling: y = alpha*x
void vector_scale(double *y, double alpha, const double *x, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        y[i] = alpha * x[i];
    }
}

// Verify solution: compute residual ||b - A*x||
double verify_solution(const double *A, const double *b, const double *x, int N) {
    double *Ax = (double*)malloc(N * sizeof(double));
    matrix_vector_multiply(A, x, Ax, N);

    double residual = 0.0;
    #pragma omp parallel for reduction(+:residual)
    for (int i = 0; i < N; i++) {
        double diff = b[i] - Ax[i];
        residual += diff * diff;
    }
    free(Ax);
    return sqrt(residual);
}

int main(int argc, char **argv) {
    // Problem size
    int N = 1024;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // Get number of OpenMP threads
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    }

    printf("OpenMP Parallel Conjugate Gradient\n");
    printf("Threads: %d\n", num_threads);
    printf("Matrix size: %d x %d\n", N, N);

    // Allocate memory
    double *A = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));
    double *r = (double*)malloc(N * sizeof(double));
    double *p = (double*)malloc(N * sizeof(double));
    double *Ap = (double*)malloc(N * sizeof(double));

    // Initialize x = 0
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
    }

    // Generate SPD matrix and RHS
    generate_spd_matrix(A, b, N);

    // CG parameters
    const double tol = 1e-6;
    const int max_iter = 10000;
    const double EPS_BREAK = 1e-30;

    double r_dot_r, r_dot_r_new, p_dot_Ap, alpha, beta;
    int iter;

    // Start timing
    double start_time = get_time();

    // Initialize: r0 = b - A*x0, but since x0 = 0, r0 = b
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        r[i] = b[i];
        p[i] = r[i]; // p0 = r0
    }

    // Initial r_dot_r = r0^T * r0
    r_dot_r = dot_product(r, r, N);
    double initial_residual = sqrt(r_dot_r);
    printf("Initial residual: %e\n", initial_residual);

    // Main CG loop
    for (iter = 0; iter < max_iter; iter++) {
        // Compute Ap = A*p
        matrix_vector_multiply(A, p, Ap, N);

        // Compute p_dot_Ap = p^T * Ap
        p_dot_Ap = dot_product(p, Ap, N);

        // Check for breakdown
        if (fabs(p_dot_Ap) < EPS_BREAK) {
            fprintf(stderr, "CG breakdown: p^T A p ~ 0 (pAp=%e) at iter %d\n", p_dot_Ap, iter);
            break;
        }

        // Compute alpha = r_dot_r / p_dot_Ap
        alpha = r_dot_r / p_dot_Ap;

        // Update x = x + alpha*p
        vector_axpy(x, x, alpha, p, N);

        // Update r = r - alpha*Ap
        vector_axpy(r, r, -alpha, Ap, N);

        // Compute new r_dot_r = r^T * r
        r_dot_r_new = dot_product(r, r, N);

        // Check convergence
        if (sqrt(r_dot_r_new) < tol * initial_residual) {
            printf("Converged after %d iterations\n", iter + 1);
            ++iter;
            r_dot_r = r_dot_r_new;
            break;
        }

        // Compute beta = r_dot_r_new / r_dot_r
        beta = r_dot_r_new / r_dot_r;

        // Update p = r + beta*p
        vector_axpy(p, r, beta, p, N);

        // Update r_dot_r for next iteration
        r_dot_r = r_dot_r_new;
    }

    // End timing
    double end_time = get_time();
    double elapsed = end_time - start_time;

    // Verify solution
    double final_residual = verify_solution(A, b, x, N);

    // Print results
    printf("Iterations: %d\n", iter);
    printf("Final residual: %e\n", sqrt(r_dot_r));
    printf("Verification residual: %e\n", final_residual);
    printf("Total time: %.6f seconds\n", elapsed);

    // Clean up
    free(A);
    free(b);
    free(x);
    free(r);
    free(p);
    free(Ap);

    return 0;
}
