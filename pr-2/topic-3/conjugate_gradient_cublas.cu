#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                             \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CHECK_CUBLAS(call)                                                        \
    do {                                                                          \
        cublasStatus_t status = (call);                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                           \
                    __FILE__, __LINE__, status);                                  \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Function to get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Function to generate a symmetric positive-definite matrix
void generate_spd_matrix(double *A, double *b, int N) {
    int i, j;
    srand(42);  // Fixed seed for reproducibility

    // First generate a random matrix
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            // Generate random value between -0.5 and 0.5
            A[i*N + j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    // Make it symmetric: A = 0.5 * (A + A^T)
    for (i = 0; i < N; i++) {
        for (j = 0; j < i; j++) {  // Only need to process lower triangle
            double avg = (A[i*N + j] + A[j*N + i]) * 0.5;
            A[i*N + j] = A[j*N + i] = avg;
        }
    }

    // Make it diagonally dominant to ensure positive definiteness
    for (i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (j = 0; j < N; j++) {
            if (i != j) {
                row_sum += fabs(A[i*N + j]);
            }
        }
        // Make diagonal elements larger than sum of other elements in row
        A[i*N + i] = row_sum + 1.0;
    }

    // Generate right-hand side vector b
    for (i = 0; i < N; i++) {
        b[i] = ((double)rand() / RAND_MAX) * 10.0;
    }
}

// Function for matrix-vector multiplication (y = A*x) using cuBLAS
void matrix_vector_multiply(cublasHandle_t handle, double *d_A, double *d_x, double *d_y, int N) {
    double alpha = 1.0;
    double beta = 0.0;

    // cublasDgemv: C = alpha*op(A)*x + beta*C
    // CUBLAS_OP_N: A is not transposed
    cublasDgemv(handle, CUBLAS_OP_N, N, N, &alpha, d_A, N, d_x, 1, &beta, d_y, 1);
}

// Function to verify the solution using CPU
double verify_solution(double *A, double *x, double *b, int N) {
    double *residual = (double*)malloc(N * sizeof(double));

    // Compute residual: r = b - A*x
    for (int i = 0; i < N; i++) {
        residual[i] = b[i];
        for (int j = 0; j < N; j++) {
            residual[i] -= A[i*N + j] * x[j];
        }
    }

    // Compute norm ||r||
    double norm = 0.0;
    for (int i = 0; i < N; i++) {
        norm += residual[i] * residual[i];
    }

    free(residual);
    return sqrt(norm);
}

int main(int argc, char *argv[]) {
    // Default parameters
    int N = 1000;        // Matrix size
    int max_iter = 1000;  // Maximum iterations
    double tol = 1e-6;    // Tolerance

    // Parse command line arguments
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) max_iter = atoi(argv[2]);
    if (argc > 3) tol = atof(argv[3]);

    printf("Conjugate Gradient Method (cuBLAS)\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Maximum iterations: %d\n", max_iter);
    printf("Tolerance: %e\n", tol);

    // Allocate host memory
    double *A = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));

    if (!A || !b || !x) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // Generate problem
    generate_spd_matrix(A, b, N);

    // Initialize solution
    for (int i = 0; i < N; i++) x[i] = 0.0;

    // Allocate device memory
    double *d_A, *d_b, *d_x, *d_r, *d_p, *d_Ap;

    CHECK_CUDA(cudaMalloc((void**)&d_A, N * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_r, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_p, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_Ap, N * sizeof(double)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice));

    // Setup cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Timing variables
    double start_time, end_time;
    double compute_time = 0.0;
    double comm_time = 0.0;

    // Variables for CG algorithm
    double alpha, beta;
    double r_dot_r, r_dot_r_new, p_dot_Ap;
    int iter;

    // Start timer
    start_time = get_time();
    double compute_start, compute_end, comm_start, comm_end;

    // Initialize: r0 = b - A*x0, but since x0 = 0, r0 = b
    comm_start = get_time();
    CHECK_CUDA(cudaMemcpy(d_r, d_b, N * sizeof(double), cudaMemcpyDeviceToDevice));
    // p0 = r0
    CHECK_CUDA(cudaMemcpy(d_p, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice));
    comm_end = get_time();
    comm_time += comm_end - comm_start;

    // Initial r_dot_r = r0^T * r0
    compute_start = get_time();
    CHECK_CUBLAS(cublasDdot(handle, N, d_r, 1, d_r, 1, &r_dot_r));
    compute_end = get_time();
    compute_time += compute_end - compute_start;

    double initial_residual = sqrt(r_dot_r);
    printf("Initial residual: %e\n", initial_residual);

    const double EPS_BREAK = 1e-30;

    // Main CG loop
    for (iter = 0; iter < max_iter; iter++) {
        // Compute Ap = A*p
        compute_start = get_time();
        matrix_vector_multiply(handle, d_A, d_p, d_Ap, N);

        // Compute p_dot_Ap = p^T * Ap
        CHECK_CUBLAS(cublasDdot(handle, N, d_p, 1, d_Ap, 1, &p_dot_Ap));

        if (fabs(p_dot_Ap) < EPS_BREAK) {
            fprintf(stderr, "CG breakdown: p^T A p ~ 0 (pAp=%e) at iter %d\n", p_dot_Ap, iter);
            break;
        }

        // Compute alpha = r_dot_r / p_dot_Ap
        alpha = r_dot_r / p_dot_Ap;
        compute_end = get_time();
        compute_time += compute_end - compute_start;

        // Update x = x + alpha*p
        compute_start = get_time();
        double alpha_pos = alpha;
        CHECK_CUBLAS(cublasDaxpy(handle, N, &alpha_pos, d_p, 1, d_x, 1));
        compute_end = get_time();
        compute_time += compute_end - compute_start;

        // Update r = r - alpha*Ap
        compute_start = get_time();
        double alpha_neg = -alpha;
        CHECK_CUBLAS(cublasDaxpy(handle, N, &alpha_neg, d_Ap, 1, d_r, 1));
        compute_end = get_time();
        compute_time += compute_end - compute_start;

        // Check convergence
        compute_start = get_time();
        CHECK_CUBLAS(cublasDdot(handle, N, d_r, 1, d_r, 1, &r_dot_r_new));

        if (sqrt(r_dot_r_new) < tol * initial_residual) {
            printf("Converged after %d iterations\n", iter + 1);
            ++iter;
            r_dot_r = r_dot_r_new;
            break;
        }
        compute_end = get_time();
        compute_time += compute_end - compute_start;

        // Compute beta = r_dot_r_new / r_dot_r
        compute_start = get_time();
        beta = r_dot_r_new / r_dot_r;

        // Update p = r + beta*p (first scale p by beta, then add r)
        CHECK_CUBLAS(cublasDscal(handle, N, &beta, d_p, 1));
        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(handle, N, &one, d_r, 1, d_p, 1));

        // Update r_dot_r for next iteration
        r_dot_r = r_dot_r_new;
        compute_end = get_time();
        compute_time += compute_end - compute_start;

        // Print progress periodically
        if ((iter + 1) % 100 == 0) {
            printf("Iteration %d: residual = %e\n", iter + 1, sqrt(r_dot_r_new));
        }
    }

    // Copy result back to host
    comm_start = get_time();
    CHECK_CUDA(cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    comm_end = get_time();
    comm_time += comm_end - comm_start;

    // Stop timer
    end_time = get_time();
    double total_time = end_time - start_time;

    // Verify solution
    double residual_norm = verify_solution(A, x, b, N);

    // Print results
    printf("\n--- Results ---\n");
    printf("Final residual norm: %e\n", residual_norm);
    printf("Iterations: %d\n", iter);
    printf("Total time: %f seconds\n", total_time);
    printf("Compute time: %f seconds\n", compute_time);
    printf("Communication time: %f seconds\n", comm_time);
    printf("Compute/Comm ratio: %f\n", compute_time / comm_time);

    // Clean up
    free(A);
    free(b);
    free(x);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_p));
    CHECK_CUDA(cudaFree(d_Ap));

    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}