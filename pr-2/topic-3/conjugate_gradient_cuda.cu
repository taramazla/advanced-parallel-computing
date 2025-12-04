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

// Function to get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Function to generate a symmetric positive-definite matrix
void generate_spd_matrix(double *A, double *b, int N) {
    int i, j;
    srand(12345);  // Fixed seed for reproducibility

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i*N + j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < i; j++) {
            double avg = (A[i*N + j] + A[j*N + i]) * 0.5;
            A[i*N + j] = A[j*N + i] = avg;
        }
    }

    for (i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (j = 0; j < N; j++) {
            if (i != j) row_sum += fabs(A[i*N + j]);
        }
        A[i*N + i] = row_sum + 1.0;
    }

    for (i = 0; i < N; i++) {
        b[i] = ((double)rand() / RAND_MAX) * 10.0;
    }
}

// CUDA kernel for matrix-vector multiplication (y = A*x)
__global__ void matrix_vector_multiply(double *A, double *x, double *y, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int row = tid; row < N; row += stride) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[row * N + j] * x[j];
        }
        y[row] = sum;
    }
}

// CUDA kernel for vector addition (a = b + alpha*c)
__global__ void vector_add(double *a, double *b, double *c, double alpha, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < N; idx += stride) {
        a[idx] = b[idx] + alpha * c[idx];
    }
}

// CUDA kernel for vector subtraction (a = b - alpha*c)
__global__ void vector_subtract(double *a, double *b, double *c, double alpha, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < N; idx += stride) {
        a[idx] = b[idx] - alpha * c[idx];
    }
}

// CUDA kernel for vector scaling (y = alpha*x)
__global__ void vector_scale(double *y, double *x, double alpha, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < N; idx += stride) {
        y[idx] = alpha * x[idx];
    }
}

// Host function for computing dot product (using cuBLAS for simplicity/performance)
double dot_product(double *d_a, double *d_b, int N, cublasHandle_t handle) {
    double result;
    cublasDdot(handle, N, d_a, 1, d_b, 1, &result);
    return result;
}

// Function to verify the solution using CPU
double verify_solution(double *A, double *x, double *b, int N) {
    double *residual = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        residual[i] = b[i];
        for (int j = 0; j < N; j++) {
            residual[i] -= A[i*N + j] * x[j];
        }
    }

    double norm = 0.0;
    for (int i = 0; i < N; i++) {
        norm += residual[i] * residual[i];
    }

    free(residual);
    return sqrt(norm);
}

int main(int argc, char *argv[]) {
    int N = 1000;
    int max_iter = 1000;
    double tol = 1e-6;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) max_iter = atoi(argv[2]);
    if (argc > 3) tol = atof(argv[3]);

    printf("Conjugate Gradient Method (CUDA Custom Kernels)\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Maximum iterations: %d\n", max_iter);
    printf("Tolerance: %e\n", tol);

    double *A = (double*)malloc(N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));

    if (!A || !b || !x) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    generate_spd_matrix(A, b, N);
    for (int i = 0; i < N; i++) x[i] = 0.0;

    double *d_A, *d_b, *d_x, *d_r, *d_p, *d_Ap;

    CHECK_CUDA(cudaMalloc((void**)&d_A, N * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_r, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_p, N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_Ap, N * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cublasHandle_t handle;
    cublasCreate(&handle);

    double start_time, end_time;
    double compute_time = 0.0;
    double comm_time = 0.0;

    double alpha, beta;
    double r_dot_r, r_dot_r_new, p_dot_Ap;
    int iter;

    start_time = get_time();
    double compute_start, compute_end, comm_start, comm_end;

    // Initialize: r0 = b - A*x0 => r0 = b
    comm_start = get_time();
    CHECK_CUDA(cudaMemcpy(d_r, d_b, N * sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_p, d_r, N * sizeof(double), cudaMemcpyDeviceToDevice));
    comm_end = get_time();
    comm_time += comm_end - comm_start;

    // Initial r_dot_r
    compute_start = get_time();
    r_dot_r = dot_product(d_r, d_r, N, handle);
    compute_end = get_time();
    compute_time += compute_end - compute_start;

    double initial_residual = sqrt(r_dot_r);
    printf("Initial residual: %e\n", initial_residual);

    const double EPS_BREAK = 1e-30;

    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A*p
        compute_start = get_time();
        matrix_vector_multiply<<<gridSize, blockSize>>>(d_A, d_p, d_Ap, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // p_dot_Ap = p^T * Ap
        p_dot_Ap = dot_product(d_p, d_Ap, N, handle);

        if (fabs(p_dot_Ap) < EPS_BREAK) {
            fprintf(stderr, "CG breakdown: p^T A p ~ 0 (pAp=%e) at iter %d\n", p_dot_Ap, iter);
            break;
        }

        // alpha = r_dot_r / p_dot_Ap
        alpha = r_dot_r / p_dot_Ap;
        compute_end = get_time();
        compute_time += compute_end - compute_start;

        // x = x + alpha*p
        compute_start = get_time();
        vector_add<<<gridSize, blockSize>>>(d_x, d_x, d_p, alpha, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        compute_end = get_time();
        compute_time += compute_end - compute_start;

        // r = r - alpha*Ap
        compute_start = get_time();
        vector_subtract<<<gridSize, blockSize>>>(d_r, d_r, d_Ap, alpha, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        compute_end = get_time();
        compute_time += compute_end - compute_start;

        // Check convergence
        compute_start = get_time();
        r_dot_r_new = dot_product(d_r, d_r, N, handle);

        if (sqrt(r_dot_r_new) < tol * initial_residual) {
            printf("Converged after %d iterations\n", iter + 1);
            break;
        }
        compute_end = get_time();
        compute_time += compute_end - compute_start;

        // beta = r_dot_r_new / r_dot_r
        compute_start = get_time();
        beta = r_dot_r_new / r_dot_r;

        // p = r + beta*p
        vector_add<<<gridSize, blockSize>>>(d_p, d_r, d_p, beta, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        r_dot_r = r_dot_r_new;
        compute_end = get_time();
        compute_time += compute_end - compute_start;

        if ((iter + 1) % 100 == 0) {
            printf("Iteration %d: residual = %e\n", iter + 1, sqrt(r_dot_r_new));
        }
    }

    comm_start = get_time();
    CHECK_CUDA(cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    comm_end = get_time();
    comm_time += comm_end - comm_start;

    end_time = get_time();
    double total_time = end_time - start_time;
    double residual_norm = verify_solution(A, x, b, N);

    printf("\n--- Results ---\n");
    printf("Final residual norm: %e\n", residual_norm);
    printf("Iterations: %d\n", iter);
    printf("Total time: %f seconds\n", total_time);
    printf("Compute time: %f seconds\n", compute_time);
    printf("Communication time: %f seconds\n", comm_time);
    printf("Compute/Comm ratio: %f\n", compute_time / comm_time);

    free(A); free(b); free(x);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_p));
    CHECK_CUDA(cudaFree(d_Ap));
    cublasDestroy(handle);

    return 0;
}