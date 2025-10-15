#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> // Required for srand

#define N 256
#define MAX_ITER 1000
#define TOLERANCE 1e-6
// #define CHECK_RESIDUAL_FREQUENCY 10 // Not currently used

// Stores the local part of the matrix and vector
typedef struct {
  double **local_A; // Pointer to rows owned by this process (local_rows x N)
  double
      *local_b; // Pointer to vector elements owned by this process (local_rows)
  int local_rows; // Number of rows owned by this process
  int start_row;  // Global index of the first row owned by this process
} LocalMatrix;

// Initializes the local part of a SYMMETRIC diagonally dominant matrix A and
// vector b This version generates the full matrix conceptually first to ensure
// symmetry robustly.
void initialize_local_symmetric_matrix(LocalMatrix *lm, int n_global,
                                       int local_rows, int start_row, int rank,
                                       int size) {
  lm->local_rows = local_rows;
  lm->start_row = start_row;

  // Allocate local storage
  lm->local_A = (double **)malloc(local_rows * sizeof(double *));
  if (!lm->local_A) {
    fprintf(stderr, "Rank %d: Failed local A alloc rows\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  for (int i = 0; i < local_rows; i++) {
    lm->local_A[i] = (double *)malloc(n_global * sizeof(double));
    if (!lm->local_A[i]) {
      fprintf(stderr, "Rank %d: Failed local A alloc row %d\n", rank, i);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  lm->local_b = (double *)malloc(local_rows * sizeof(double));
  if (!lm->local_b) {
    fprintf(stderr, "Rank %d: Failed local b alloc\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // --- Generate full matrix conceptually on each process ---
  double **A_full = NULL;
  double *b_full = NULL;

  if (rank == 0)
    printf("Initializing matrix (using temporary full matrix)...\n");

  A_full = (double **)malloc(n_global * sizeof(double *));
  b_full = (double *)malloc(n_global * sizeof(double));
  if (!A_full || !b_full) {
    fprintf(stderr, "Rank %d: Failed temp full matrix alloc\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  for (int i = 0; i < n_global; ++i) {
    A_full[i] = (double *)malloc(n_global * sizeof(double));
    if (!A_full[i]) {
      fprintf(stderr, "Rank %d: Failed temp full matrix row alloc %d\n", rank,
              i);
      for (int k = 0; k < i; ++k)
        free(A_full[k]); // Basic cleanup
      free(A_full);
      free(b_full);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int j = 0; j < n_global; ++j)
      A_full[i][j] = 0.0; // Initialize
  }

  srand(0037); // Fixed seed

  // 1. Generate upper triangle
  for (int i = 0; i < n_global; ++i) {
    for (int j = i; j < n_global; ++j) {
      if (i != j)
        A_full[i][j] = 20.0 * (rand() / (double)RAND_MAX) - 10.0;
    }
  }

  // 2. Enforce symmetry
  for (int i = 0; i < n_global; ++i) {
    for (int j = 0; j < i; ++j)
      A_full[i][j] = A_full[j][i];
  }

  // 3. Set diagonal and generate b
  srand(0037);                         // Reset seed state
  for (int i = 0; i < n_global; ++i) { // Advance rand() for off-diagonals first
    for (int j = i + 1; j < n_global; ++j) {
      rand();
    }
  }
  for (int i = 0; i < n_global; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n_global; ++j)
      if (i != j)
        row_sum += fabs(A_full[i][j]);
    A_full[i][i] =
        row_sum + 5.0 + 10.0 * (rand() / (double)RAND_MAX); // Set diagonal
    b_full[i] = 20.0 * (rand() / (double)RAND_MAX) - 10.0;  // Generate b
  }

  // --- Copy local parts ---
  for (int local_i = 0; local_i < local_rows; ++local_i) {
    int global_i = start_row + local_i;
    for (int j = 0; j < n_global; ++j)
      lm->local_A[local_i][j] = A_full[global_i][j];
    lm->local_b[local_i] = b_full[global_i];
  }

  // --- Free temporary full matrix/vector ---
  for (int i = 0; i < n_global; ++i)
    if (A_full[i])
      free(A_full[i]);
  free(A_full);
  free(b_full);
  if (rank == 0)
    printf("Matrix initialization complete.\n");
}

void free_local_matrix(LocalMatrix *lm) {
  if (!lm)
    return;
  if (lm->local_A) {
    for (int i = 0; i < lm->local_rows; i++)
      free(lm->local_A[i]);
    free(lm->local_A);
    lm->local_A = NULL;
  }
  free(lm->local_b);
  lm->local_b = NULL;
}

// Parallel Dot Product
double parallel_dot_product(const double *v_local, const double *w_local,
                            int local_n) {
  double local_dot = 0.0;
  for (int i = 0; i < local_n; ++i)
    local_dot += v_local[i] * w_local[i];
  double global_dot = 0.0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  return global_dot;
}

// Parallel Matrix-Vector Product
void parallel_matvec(double *Ap_local, const LocalMatrix *lm,
                     const double *p_global) {
  for (int i = 0; i < lm->local_rows; ++i) {
    Ap_local[i] = 0.0;
    for (int j = 0; j < N; ++j)
      Ap_local[i] += lm->local_A[i][j] * p_global[j];
  }
}

// Parallel Vector Update: y = y + a*x
void parallel_axpy(double a, const double *x_local, double *y_local,
                   int local_n) {
  for (int i = 0; i < local_n; ++i)
    y_local[i] += a * x_local[i];
}

// Parallel Vector Update: y = x + a*y
void parallel_y_eq_x_plus_ay(const double *x_local, double a, double *y_local,
                             int local_n) {
  for (int i = 0; i < local_n; ++i)
    y_local[i] = x_local[i] + a * y_local[i];
}

int main(int argc, char **argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double start_time;
  double comm_start = 0.0, comp_start = 0.0;
  double init_time = 0.0, total_comm_time = 0.0, total_comp_time = 0.0,
         total_time = 0.0;

  start_time = MPI_Wtime();

  // --- Distribution ---
  int rows_per_proc = N / size;
  int remainder = N % size;
  int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
  int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);

  // --- Initialization ---
  double init_start = MPI_Wtime();

  // Allocate local vectors
  double *x_local = (double *)calloc(local_rows, sizeof(double));
  double *r_local = (double *)malloc(local_rows * sizeof(double));
  double *p_local = (double *)malloc(local_rows * sizeof(double));
  double *Ap_local = (double *)malloc(local_rows * sizeof(double));
  double *Ax_final_local =
      (double *)malloc(local_rows * sizeof(double)); // For verification

  // Allocate full vectors
  double *p_global = (double *)malloc(N * sizeof(double));
  double *x_global = (double *)malloc(N * sizeof(double)); // For verification

  if (!x_local || !r_local || !p_local || !Ap_local || !p_global || !x_global ||
      !Ax_final_local) {
    fprintf(stderr, "Rank %d: Failed vector allocation\n", rank);
    // Basic cleanup before abort
    free(x_local);
    free(r_local);
    free(p_local);
    free(Ap_local);
    free(p_global);
    free(x_global);
    free(Ax_final_local);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  LocalMatrix matrix;
  initialize_local_symmetric_matrix(&matrix, N, local_rows, start_row, rank,
                                    size);

  // Initial setup: x = 0, r = b, p = r
  for (int i = 0; i < local_rows; ++i) {
    r_local[i] = matrix.local_b[i];
    p_local[i] = r_local[i];
  }

  // --- MPI_Allgatherv setup ---
  int *recvcounts = (int *)malloc(size * sizeof(int));
  int *displs = (int *)malloc(size * sizeof(int));
  if (!recvcounts || !displs) { /* Error handling */
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  for (int i = 0; i < size; i++) {
    recvcounts[i] = rows_per_proc + (i < remainder ? 1 : 0);
    displs[i] = i * rows_per_proc + (i < remainder ? i : remainder);
  }

  init_time = MPI_Wtime() - init_start;

  // --- Conjugate Gradient Iteration Loop ---
  double rsold = 0.0, rsnew = 0.0, alpha = 0.0, beta = 0.0, pTAp = 0.0;
  int iter;
  int iterations_completed = 0;

  comm_start = MPI_Wtime();
  rsold = parallel_dot_product(r_local, r_local, local_rows);
  total_comm_time += MPI_Wtime() - comm_start;

  if (sqrt(rsold) < TOLERANCE) {
    if (rank == 0)
      printf("Initial guess is already solution, residual norm %e\n",
             sqrt(rsold));
    iter = 0;
    rsnew = rsold;
  } else {
    for (iter = 0; iter < MAX_ITER; iter++) {
      iterations_completed = iter + 1;

      // 1. Gather p
      comm_start = MPI_Wtime();
      MPI_Allgatherv(p_local, local_rows, MPI_DOUBLE, p_global, recvcounts,
                     displs, MPI_DOUBLE, MPI_COMM_WORLD);
      total_comm_time += MPI_Wtime() - comm_start;

      // 2. Ap = A * p
      comp_start = MPI_Wtime();
      parallel_matvec(Ap_local, &matrix, p_global);
      total_comp_time += MPI_Wtime() - comp_start;

      // 3. alpha = rsold / (p' * Ap)
      comm_start = MPI_Wtime();
      pTAp = parallel_dot_product(p_local, Ap_local, local_rows);
      total_comm_time += MPI_Wtime() - comm_start;

      if (pTAp < 1e-12) {
        if (rank == 0) {
          if (pTAp <= 0)
            fprintf(stderr,
                    "CG breakdown: p'Ap = %e <= 0. Matrix N-PD? Iter %d\n",
                    pTAp, iter);
          else
            fprintf(stderr, "CG breakdown: p'Ap = %e is too small at iter %d\n",
                    pTAp, iter);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        rsnew = parallel_dot_product(r_local, r_local,
                                     local_rows); // Use current residual
        iter = MAX_ITER;                          // Force exit indication
        break;
      }
      alpha = rsold / pTAp;

      // 4. x = x + alpha * p
      comp_start = MPI_Wtime();
      parallel_axpy(alpha, p_local, x_local, local_rows);
      total_comp_time += MPI_Wtime() - comp_start;

      // 5. r = r - alpha * Ap
      comp_start = MPI_Wtime();
      parallel_axpy(-alpha, Ap_local, r_local, local_rows);
      total_comp_time += MPI_Wtime() - comp_start;

      // 6. rsnew = r' * r
      comm_start = MPI_Wtime();
      rsnew = parallel_dot_product(r_local, r_local, local_rows);
      total_comm_time += MPI_Wtime() - comm_start;

      // 7. Check convergence
      if (sqrt(rsnew) < TOLERANCE)
        break;

      if (fabs(rsold) < 1e-18) { // Check before division
        if (rank == 0)
          fprintf(stderr, "CG Warning: rsold = %e is very small at iter %d.\n",
                  rsold, iter);
      }

      // 8. beta = rsnew / rsold
      beta = rsnew / rsold;

      // 9. p = r + beta * p
      comp_start = MPI_Wtime();
      parallel_y_eq_x_plus_ay(r_local, beta, p_local, local_rows);
      total_comp_time += MPI_Wtime() - comp_start;

      // 10. Update rsold
      rsold = rsnew;
    } // End of iteration loop
  } // End else block

  // --- Verification Step ---
  double verification_residual_norm = -1.0; // Init verification value

  // 1. Gather final solution x
  comm_start = MPI_Wtime();
  MPI_Allgatherv(x_local, local_rows, MPI_DOUBLE, x_global, recvcounts, displs,
                 MPI_DOUBLE, MPI_COMM_WORLD);
  total_comm_time += MPI_Wtime() - comm_start; // Include gather in comm time

  // 2. Compute Ax_final = A * x_final (local part)
  comp_start = MPI_Wtime();
  parallel_matvec(Ax_final_local, &matrix, x_global);
  total_comp_time += MPI_Wtime() - comp_start; // Include matvec in comp time

  // 3. Compute final residual r_final = b - Ax_final (local part)
  // Reuse r_local buffer for this calculation
  comp_start = MPI_Wtime();
  for (int i = 0; i < local_rows; ++i) {
    r_local[i] = matrix.local_b[i] - Ax_final_local[i];
  }
  total_comp_time +=
      MPI_Wtime() - comp_start; // Include residual calc in comp time

  // 4. Compute norm ||r_final||
  comm_start = MPI_Wtime();
  double verification_residual_norm_sq =
      parallel_dot_product(r_local, r_local, local_rows);
  verification_residual_norm = sqrt(verification_residual_norm_sq);
  total_comm_time +=
      MPI_Wtime() - comm_start; // Include dot product comm in comm time

  total_time = MPI_Wtime() - start_time; // Total execution time

  // --- Collect and Print Results (Rank 0) ---
  double max_comp_time = 0.0, avg_comm_time = 0.0, max_init_time = 0.0,
         max_total_time = 0.0;

  MPI_Reduce(&total_comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&total_comm_time, &avg_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&init_time, &max_init_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    double final_cg_residual_norm =
        sqrt(rsnew); // Residual norm from the *end* of the CG loop
    if (iter >= MAX_ITER) {
      printf("\nWarning: CG failed to converge or broke down within %d "
             "iterations.\n",
             MAX_ITER);
      printf("Final CG residual norm ||r_k|| = %e\n", final_cg_residual_norm);
    } else {
      printf("\nCG converged in %d iterations.\n", iterations_completed);
      printf("Final CG residual norm ||r_k|| = %e (Tolerance: %e)\n",
             final_cg_residual_norm, TOLERANCE);
    }

    // Print Verification Result
    printf("--- Verification ---\n");
    printf("Computed ||b - Ax|| = %e\n", verification_residual_norm);
    // Allow slightly larger tolerance for verification due to potential FP
    // differences
    if (verification_residual_norm <= TOLERANCE * 100) {
      printf("Verification PASSED (||b - Ax|| <= Tolerance * 100)\n");
    } else {
      printf("Verification FAILED (Residual norm ||b - Ax|| too high)\n");
    }
    printf("--------------------\n");

    avg_comm_time /= size;

    printf("\n=========== Performance Results ===========\n");
    printf("Matrix Size (N):          %d\n", N);
    printf("Number of Processes:      %d\n", size);
    printf("Max Initialization Time:  %.6f seconds\n", max_init_time);
    printf("Avg Communication Time:   %.6f seconds\n",
           avg_comm_time); // Includes verification comm
    printf("Max Computation Time:     %.6f seconds\n",
           max_comp_time); // Includes verification comp
    printf("Max Total Execution Time: %.6f seconds\n", max_total_time);
    printf("-------------------------------------------\n");
    if (max_total_time > 1e-9) {
      // Note: Percentages now include verification time contributions
      printf("Computation Percentage (Max): %.2f%%\n",
             100.0 * max_comp_time / max_total_time);
      printf("Communication Percentage (Avg): %.2f%%\n",
             100.0 * avg_comm_time / max_total_time);
    }

    // FLOPS calculation (Approximate - only for CG iterations)
    double flops_matvec = 2.0 * N * N;
    double flops_dotprod = 2.0 * N;
    double flops_axpy = 2.0 * N;
    double total_cg_flops =
        (double)iterations_completed *
        (flops_matvec + 2.0 * flops_dotprod + 3.0 * flops_axpy);

    // Calculate GFLOPS based on CG computation time *only* if possible
    // Subtract approximate verification computation time from max_comp_time
    double approx_verify_comp_flops =
        (2.0 * N * N) + (2.0 * N) +
        (2.0 * N); // Matvec + residual calc + dotprod local
    // This is a rough estimate, requires timing verification separately for
    // accuracy For now, report GFLOPS based on total computation time including
    // verification
    if (max_comp_time > 1e-9) {
      double gflops = total_cg_flops / (max_comp_time * 1e9);
      printf("Performance (Based on Max Comp Time, incl. verification): %.4f "
             "GFLOPS\n",
             gflops);
    } else {
      printf("Performance (Based on Max Comp Time): N/A\n");
    }
    printf("===========================================\n");
  }

  // --- Cleanup ---
  free(x_local);
  free(r_local); // Reused for verification residual
  free(p_local);
  free(Ap_local);
  free(Ax_final_local); // Free verification vector
  free(p_global);
  free(x_global); // Free verification vector
  free(recvcounts);
  free(displs);
  free_local_matrix(&matrix);

  MPI_Finalize();
  return 0;
}
