// conjugateGradient.cu
// Single-file implementation of Conjugate Gradient on CUDA
// Inspired by Tim Lebailly's implementation, but self-contained in one .cu file.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <ctime>
#include <cstdint>
#include <sys/time.h>
#include <sys/stat.h>

#include <cuda_runtime.h>

// =====================
//  Macros & constants
// =====================

// vecVec
#define BLOCK_DIM_VEC 1024

// matVec
#define NB_ELEM_MAT 32
#define BLOCK_SIZE_MAT 32

#define LOG_FILE_FORMAT "conjugateGradient-%d.csv"

// Indexing helpers (row-major)
#define A(row, col, N) (A[(row) * (N) + (col)])
#define b(i)           (b[(i)])

// Simple CUDA error macro (no-op behavior, but keeps syntax)
#define CUDACHECK(x) (x)

// =====================
//  Helper declarations
// =====================

void parseArgs(int argc, char *argv[], int *NMin, int *NMax, int *NMult,
               int *MAX_ITER, float *EPS, float *TOL);

float *generateA(int N);
float *generateb(int N);
void printMat(float *A, int N);
void printVec(float *b, int N);
float getMaxDiffSquared(float *a, float *b, int N);
long  getMicrotime();
int   moreOrLessEqual(float *a, float *b, int N, float TOL);

// Sequential CG
void matVec_seq(float *A, float *b, float *out, int N);
float vecVec_seq(float *vec1, float *vec2, int N);
void vecPlusVec_seq(float *vec1, float *vec2, float *out, int N);
void scalarVec_seq(float alpha, float *vec2, float *out, int N);
void scalarMatVec_seq(float alpha, float *A, float *b, float *out, int N);
float norm2d_seq(float *a, int N);
void solveCG_seq(float *A, float *b, float *x, float *r_norm,
                 int *cnt, int N, int MAX_ITER, float EPS);

// CUDA CG
void solveCG_cuda(float *A, float *b, float *x, float *p, float *r, float *temp,
                  float *alpha, float *beta, float *r_norm, float *r_norm_old,
                  float *temp_scal, float *h_x, float *h_r_norm, int *cnt,
                  int N, int maxIter, float eps);

// =====================
//  CUDA kernels
// =====================

/*
 * Naive matrix-vector product: out = A * b
 * A: N x N (row-major), b: N, out: N
 */
__global__ void matVec(float *A, float *b, float *out, int N) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x < N) {
        float tmp = 0.0f;
        for (int i = 0; i < N; i++) {
            tmp += b[i] * A[index_x * N + i];
        }
        out[index_x] = tmp;
    }
}

/*
 * More efficient symmetric matrix-vector product using shared memory.
 * out += A * b  (with atomicAdd on out)
 */
__global__ void matVec2(float *A, float *b, float *out, int N) {
    __shared__ float b_shared[NB_ELEM_MAT];

    int effective_block_width;
    if ((blockIdx.x + 1) * NB_ELEM_MAT <= N) {
        effective_block_width = NB_ELEM_MAT;
    } else {
        effective_block_width = N % NB_ELEM_MAT;
    }

    // Load tile of b into shared memory
    if (threadIdx.x < effective_block_width) {
        b_shared[threadIdx.x] = b[blockIdx.x * NB_ELEM_MAT + threadIdx.x];
    }

    __syncthreads();

    int idy = blockIdx.y * BLOCK_SIZE_MAT + threadIdx.x;
    float tmp_scal = 0.0f;

    if (idy < N) {
        for (int i = 0; i < effective_block_width; i++) {
            // A is symmetric: A(row, col) = A(col, row)
            int row = blockIdx.x * NB_ELEM_MAT + i;
            int col = idy;
            tmp_scal += b_shared[i] * A[row * N + col];
        }
        atomicAdd(out + idy, tmp_scal);
    }
}

/*
 * out = a + b
 */
__global__ void vecPlusVec(float *a, float *b, float *out, int N) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x < N) {
        out[index_x] = b[index_x] + a[index_x];
    }
}

/*
 * out = a + b, and zero b
 */
__global__ void vecPlusVec2(float *a, float *b, float *out, int N) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x < N) {
        out[index_x] = b[index_x] + a[index_x];
        b[index_x] = 0.0f;
    }
}

/*
 * out = a - b
 */
__global__ void vecMinVec(float *a, float *b, float *out, int N) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x < N) {
        out[index_x] = a[index_x] - b[index_x];
    }
}

/*
 * Naive dot product (single thread)
 */
__global__ void vecVec(float *a, float *b, float *out, int N) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;
    if (index_x == 0) {
        for (int i = 0; i < N; i++) {
            tmp += b[i] * a[i];
        }
        *out = tmp;
    }
}

/*
 * Efficient dot product using shared memory + block reduction + atomicAdd.
 */
__global__ void vecVec2(float *a, float *b, float *out, int N) {
    __shared__ float shared_tmp[BLOCK_DIM_VEC];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize output once for all blocks (thread 0)
    if (gid == 0) {
        *out = 0.0f;
    }

    if (gid < N) {
        shared_tmp[threadIdx.x] = a[gid] * b[gid];
    } else {
        shared_tmp[threadIdx.x] = 0.0f;
    }

    // Block reduction
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, shared_tmp[0]);
    }
}

/*
 * out = scalar * a
 */
__global__ void scalarVec(float *scalar, float *a, float *out, int N) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x < N) {
        out[index_x] = a[index_x] * (*scalar);
    }
}

/*
 * out = in
 */
__global__ void memCopy(float *in, float *out, int N) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x < N) {
        out[index_x] = in[index_x];
    }
}

/*
 * out = num / den
 */
__global__ void divide(float *num, float *den, float *out) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x == 0) {
        *out = (*num) / (*den);
    }
}

// =====================
//  CUDA CG solver
// =====================

void solveCG_cuda(float *A, float *b, float *x, float *p, float *r, float *temp,
                  float *alpha, float *beta, float *r_norm, float *r_norm_old,
                  float *temp_scal, float *h_x, float *h_r_norm, int *cnt,
                  int N, int maxIter, float eps) {

    dim3 vec_block_dim(BLOCK_DIM_VEC);
    dim3 vec_grid_dim((N + BLOCK_DIM_VEC - 1) / BLOCK_DIM_VEC);

    dim3 mat_grid_dim((N + NB_ELEM_MAT - 1) / NB_ELEM_MAT,
                      (N + BLOCK_SIZE_MAT - 1) / BLOCK_SIZE_MAT);
    dim3 mat_block_dim(BLOCK_SIZE_MAT);

    // r_norm_old = r^T r
    vecVec2<<<vec_grid_dim, vec_block_dim>>>(r, r, r_norm_old, N);
    cudaDeviceSynchronize();

    int k = 0;

    // Initialize host copy of residual norm to something > eps
    *h_r_norm = 1.0f;
    while ((k < maxIter) && (*h_r_norm > eps)) {
        // temp = A * p
        cudaMemset(temp, 0, N * sizeof(float));
        matVec2<<<mat_grid_dim, mat_block_dim>>>(A, p, temp, N);

        // alpha = r_norm_old / (p^T temp)
        vecVec2<<<vec_grid_dim, vec_block_dim>>>(p, temp, temp_scal, N);
        divide<<<1, 1>>>(r_norm_old, temp_scal, alpha);

        // r = r - alpha * temp
        scalarVec<<<vec_grid_dim, vec_block_dim>>>(alpha, temp, temp, N);
        vecMinVec<<<vec_grid_dim, vec_block_dim>>>(r, temp, r, N);

        // x = x + alpha * p
        scalarVec<<<vec_grid_dim, vec_block_dim>>>(alpha, p, temp, N);
        vecPlusVec<<<vec_grid_dim, vec_block_dim>>>(x, temp, x, N);

        // r_norm = r^T r
        vecVec2<<<vec_grid_dim, vec_block_dim>>>(r, r, r_norm, N);
        // beta = r_norm / r_norm_old
        divide<<<1, 1>>>(r_norm, r_norm_old, beta);

        // p = r + beta * p  (using vecPlusVec2 to also zero temp vector)
        scalarVec<<<vec_grid_dim, vec_block_dim>>>(beta, p, temp, N);
        vecPlusVec2<<<vec_grid_dim, vec_block_dim>>>(r, temp, p, N);

        // r_norm_old = r_norm  (just a scalar copy)
        memCopy<<<1, 1>>>(r_norm, r_norm_old, 1);

        // Copy residual norm to host to check convergence
        cudaMemcpy(h_r_norm, r_norm, sizeof(float), cudaMemcpyDeviceToHost);

        k++;
    }

    *cnt = k;
}

// =====================
//  Argument parsing
// =====================

static void parseArgsInt(const char *str, int *out) {
    char *end = nullptr;
    long val = strtol(str, &end, 10);
    if (*end != '\0') {
        fprintf(stderr, "Failed to parse integer from '%s'\n", str);
        exit(EXIT_FAILURE);
    }
    *out = (int)val;
}

static void parseArgsFloat(const char *str, float *out) {
    char *end = nullptr;
    float val = strtof(str, &end);
    if (*end != '\0') {
        fprintf(stderr, "Failed to parse float from '%s'\n", str);
        exit(EXIT_FAILURE);
    }
    *out = val;
}

void parseArgs(int argc, char *argv[], int *NMin, int *NMax, int *NMult,
               int *MAX_ITER, float *EPS, float *TOL) {
    if (argc != 7) {
        fprintf(stderr,
                "[ERROR] Must be run with exactly 6 arguments, found %d!\n"
                "Usage: %s <NMin> <NMax> <NMult> <MAX_ITER> <EPS> <TOL>\n",
                argc - 1, argv[0]);
        exit(EXIT_FAILURE);
    }

    parseArgsInt(argv[1], NMin);
    parseArgsInt(argv[2], NMax);
    parseArgsInt(argv[3], NMult);
    parseArgsInt(argv[4], MAX_ITER);
    parseArgsFloat(argv[5], EPS);
    parseArgsFloat(argv[6], TOL);
}

// =====================
//  Main
// =====================

int main(int argc, char *argv[]) {
    int j, NMin, NMax, NMult, NIter, MAX_ITER;
    float EPS, TOL;

    parseArgs(argc, argv, &NMin, &NMax, &NMult, &MAX_ITER, &EPS, &TOL);

    struct stat buffer;
    int logId = 1;
    char logFileNameWithId[256];

    // Find next available log file name
    do {
        snprintf(logFileNameWithId, sizeof(logFileNameWithId),
                 LOG_FILE_FORMAT, logId);
        logId++;
    } while (stat(logFileNameWithId, &buffer) == 0);

    FILE *log_file = fopen(logFileNameWithId, "w");
    if (!log_file) {
        perror("Failed to open log file");
        return EXIT_FAILURE;
    }
    fprintf(log_file,
            "j,N,grid_size,block_size,is_ok,gpu_time,cpu_time,"
            "gpu_r_norm,cpu_r_norm,gpu_iter,cpu_iter,speedup\n");
    fclose(log_file);

    printf("\n----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("|          N |   gridSize |  blockSize |      isOk |         gpuTime |         cpuTime |        gpuRNorm |        cpuRNorm |   gpuIter |   cpuIter |         speedUp |\n");
    printf("|            |   (nBlock) |  (nThread) |           |            (ms) |            (ms) |                 |                 |           |           |                 |\n");
    printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");

    NIter = (int)(log10((double)NMax / (double)NMin) / log10((double)NMult)) + 1;

    for (j = 0; j < NIter; j++) {
        int N = NMin * (int)pow((double)NMult, j);

        // Host memory
        float *h_A = generateA(N);
        float *h_b = generateb(N);
        float *h_x = (float *)calloc(N, sizeof(float));
        float *h_r_norm = (float *)malloc(sizeof(float));
        *h_r_norm = 1.0f;

        int gpu_cnt = 0, cpu_cnt = 0;
        float cpu_r_norm = 0.0f;
        float gpu_elapsed_time_ms = 0.0f;
        float cpu_elapsed_time_ms = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Device memory
        float *d_A, *d_b, *d_x, *d_p, *d_r, *d_temp;
        cudaMalloc((void **)&d_A, N * N * sizeof(float));
        cudaMalloc((void **)&d_b, N * sizeof(float));
        cudaMalloc((void **)&d_x, N * sizeof(float));
        cudaMalloc((void **)&d_p, N * sizeof(float));
        cudaMalloc((void **)&d_r, N * sizeof(float));
        cudaMalloc((void **)&d_temp, N * sizeof(float));

        // Scalars on device
        float *d_beta, *d_alpha, *d_r_norm, *d_r_norm_old, *d_temp_scal;
        cudaMalloc((void **)&d_beta, sizeof(float));
        cudaMalloc((void **)&d_alpha, sizeof(float));
        cudaMalloc((void **)&d_r_norm, sizeof(float));
        cudaMalloc((void **)&d_r_norm_old, sizeof(float));
        cudaMalloc((void **)&d_temp_scal, sizeof(float));

        // Copy to device
        cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

        // Initial guess x0 = 0 => p0 = r0 = b
        cudaMemcpy(d_p, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

        // === GPU CG ===
        cudaEventRecord(start, 0);
        solveCG_cuda(d_A, d_b, d_x, d_p, d_r, d_temp,
                     d_alpha, d_beta, d_r_norm, d_r_norm_old,
                     d_temp_scal, h_x, h_r_norm, &gpu_cnt,
                     N, MAX_ITER, EPS);
        CUDACHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

        // Copy result back
        cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

        // === CPU CG (reference) ===
        float *h_x_seq = (float *)calloc(N, sizeof(float));

        cudaEventRecord(start, 0);
        solveCG_seq(h_A, h_b, h_x_seq, &cpu_r_norm, &cpu_cnt,
                    N, MAX_ITER, EPS);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);

        // Check correctness
        int resultIsOk = moreOrLessEqual(h_x, h_x_seq, N, TOL) == 1;
        int gridSize = (N + BLOCK_DIM_VEC - 1) / BLOCK_DIM_VEC;
        int blockSize = BLOCK_DIM_VEC;
        float speedup = cpu_elapsed_time_ms / gpu_elapsed_time_ms;

        // Append to CSV
        log_file = fopen(logFileNameWithId, "a");
        fprintf(log_file, "%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%.6f\n",
                j, N, gridSize, blockSize, resultIsOk, gpu_elapsed_time_ms,
                cpu_elapsed_time_ms, *h_r_norm, cpu_r_norm,
                gpu_cnt, cpu_cnt, speedup);
        fclose(log_file);

        printf("| %10d | %10d | %10d | %9d | %15.6f | %15.6f | %15.9e | %15.9e | %9d | %9d | %15.6f |\n",
               N, gridSize, blockSize, resultIsOk, gpu_elapsed_time_ms,
               cpu_elapsed_time_ms, *h_r_norm, cpu_r_norm,
               gpu_cnt, cpu_cnt, speedup);

        // Cleanup host
        free(h_A);
        free(h_b);
        free(h_x);
        free(h_r_norm);
        free(h_x_seq);

        // Cleanup device
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_p);
        cudaFree(d_r);
        cudaFree(d_temp);
        cudaFree(d_alpha);
        cudaFree(d_beta);
        cudaFree(d_r_norm);
        cudaFree(d_r_norm_old);
        cudaFree(d_temp_scal);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n");
    return 0;
}

// =====================
//  Helper implementations
// =====================

/*
 * Generates a dense PSD symmetric N x N matrix.
 */
float *generateA(int N) {
    int i, j;
    float *A = (float *)malloc(sizeof(float) * N * N);

    float temp;
    for (i = 0; i < N; i++) {
        for (j = 0; j <= i; j++) {
            temp = (float)rand();
            if (i == j) {
                A(i, j, N) = temp + N;   // make it diagonally dominant
            } else {
                A(i, j, N) = temp;
                A(j, i, N) = temp;
            }
        }
    }
    return A;
}

/*
 * Returns the time in microseconds
 */
long getMicrotime() {
    struct timeval currentTime;
    gettimeofday(&currentTime, NULL);
    return (long)currentTime.tv_sec * (long)1e6 + (long)currentTime.tv_usec;
}

/*
 * Generates a random vector of size N
 */
float *generateb(int N) {
    int i;
    float *b = (float *)malloc(sizeof(float) * N);
    for (i = 0; i < N; i++) {
        b[i] = (float)rand();
    }
    return b;
}

/*
 * Prints a square matrix
 */
void printMat(float *A, int N) {
    for (int i = 0; i < N * N; i++) {
        printf("%.3e ", A[i]);
        if ((i + 1) % N == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

/*
 * Prints a vector
 */
void printVec(float *b, int N) {
    printf("__begin_vector__\n");
    for (int i = 0; i < N; i++) {
        if (b[i] > 0) {
            printf("+%.6e\n", b[i]);
        } else {
            printf("%.6e\n", b[i]);
        }
    }
    printf("__end_vector__\n");
}

/*
 * Computes the max squared difference of 2 vectors
 */
float getMaxDiffSquared(float *a, float *bvec, int N) {
    float maxv = 0.0f;
    for (int i = 0; i < N; i++) {
        float tmp = (a[i] - bvec[i]) * (a[i] - bvec[i]);
        if (tmp > maxv) {
            maxv = tmp;
        }
    }
    return maxv;
}

// =====================
//  Sequential helpers & CG
// =====================

void matVec_seq(float *A, float *b, float *out, int N) {
    for (int j = 0; j < N; j++) {
        out[j] = 0.0f;
        for (int i = 0; i < N; i++) {
            out[j] += A(j, i, N) * b(i);
        }
    }
}

float vecVec_seq(float *vec1, float *vec2, int N) {
    float product = 0.0f;
    for (int i = 0; i < N; i++) {
        product += vec1[i] * vec2[i];
    }
    return product;
}

void vecPlusVec_seq(float *vec1, float *vec2, float *out, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = vec1[i] + vec2[i];
    }
}

void scalarVec_seq(float alpha, float *vec2, float *out, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = alpha * vec2[i];
    }
}

void scalarMatVec_seq(float alpha, float *A, float *b, float *out, int N) {
    for (int j = 0; j < N; j++) {
        out[j] = 0.0f;
        for (int i = 0; i < N; i++) {
            out[j] += alpha * A(j, i, N) * b(i);
        }
    }
}

float norm2d_seq(float *a, int N) {
    return sqrtf(vecVec_seq(a, a, N));
}

int moreOrLessEqual(float *a, float *bvec, int N, float TOL) {
    for (int i = 0; i < N; i++) {
        if (fabsf(a[i] - bvec[i]) > TOL) {
            return 0;
        }
    }
    return 1;
}

/*
 * Sequential Conjugate Gradient: solves Ax = b
 */
void solveCG_seq(float *A, float *bvec, float *x, float *r_norm,
                 int *cnt, int N, int MAX_ITER, float EPS) {

    float *p    = (float *)calloc(N, sizeof(float));
    float *r    = (float *)calloc(N, sizeof(float));
    float *temp = (float *)calloc(N, sizeof(float));

    float beta, alpha, rNormOld = 0.0f;
    float rNorm = 1.0f;
    int k = 0;

    // r = b - A x ; p = r
    scalarMatVec_seq(-1.0f, A, x, temp, N);
    vecPlusVec_seq(bvec, temp, r, N);
    scalarVec_seq(1.0f, r, p, N);
    rNormOld = vecVec_seq(r, r, N);

    while ((rNorm > EPS) && (k < MAX_ITER)) {
        // temp = A p
        matVec_seq(A, p, temp, N);
        // alpha = r^T r / (p^T A p)
        alpha = rNormOld / vecVec_seq(p, temp, N);
        // r = r - alpha * A p
        scalarVec_seq(-alpha, temp, temp, N);
        vecPlusVec_seq(r, temp, r, N);
        // x = x + alpha p
        scalarVec_seq(alpha, p, temp, N);
        vecPlusVec_seq(x, temp, x, N);
        // beta = r_{k+1}^T r_{k+1} / r_k^T r_k
        rNorm = vecVec_seq(r, r, N);
        beta = rNorm / rNormOld;
        // p = r + beta p
        scalarVec_seq(beta, p, temp, N);
        vecPlusVec_seq(r, temp, p, N);
        rNormOld = rNorm;
        k++;
    }

    *cnt = k;
    *r_norm = rNorm;

    free(p);
    free(r);
    free(temp);
}
