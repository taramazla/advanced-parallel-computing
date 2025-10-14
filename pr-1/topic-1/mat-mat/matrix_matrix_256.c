#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "mpi.h"

#define N 256  // number of rows and columns in matrix

MPI_Status status;

double a[N][N], b[N][N], c[N][N];

void transpose(int size, double matrix[size][size]);

int main(int argc, char **argv) {
    int numtasks, taskid, numworkers, source, dest, rows, offset, i, j, k;

    double start, stop;
    double comm_start, comm_end;
    double send_comm_time = 0.0, recv_comm_time = 0.0, total_comm_time = 0.0;
    double comp_start, comp_end, comp_time = 0.0;
    double max_comp_time, min_comp_time, avg_comp_time, sum_comp_time;
    double worker_comm_time = 0.0, max_worker_comm_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    numworkers = numtasks - 1;

    /*---------------------------- MASTER ----------------------------*/
    if (taskid == 0) {
        // Initialize matrix A and B
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                a[i][j] = 1.0;
                b[i][j] = 2.0;
            }
        }

        // Transpose matrix B for efficient column access during multiplication
        transpose(N, b);

        // Calculate rows per worker
        rows = (numworkers > 0) ? N / numworkers : N;

        // Synchronize all processes before starting timing
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();

        offset = 0;

        // Measure time for sending data to workers
        comm_start = MPI_Wtime();

        // Send data to worker processes
        for (dest = 1; dest <= numworkers; dest++) {
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&b, N * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            offset += rows;
        }

        comm_end = MPI_Wtime();
        send_comm_time = comm_end - comm_start;

        // Measure time for receiving results from workers
        comm_start = MPI_Wtime();

        // Receive results from worker processes
        for (i = 1; i <= numworkers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows * N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }

        comm_end = MPI_Wtime();
        recv_comm_time = comm_end - comm_start;

        // Calculate total communication time
        total_comm_time = send_comm_time + recv_comm_time;

        // If no workers, compute the multiplication locally
        if (numworkers == 0) {
            comp_start = MPI_Wtime();
            for (i = 0; i < rows; i++) {
                for (j = 0; j < N; j++) {
                    c[i][j] = 0;
                    for (k = 0; k < N; k++) {
                        c[i][j] += a[i][k] * b[j][k];
                    }
                }
            }
            comp_end = MPI_Wtime();
            comp_time = comp_end - comp_start;
        }

        // Collect computation times from all processes
        MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comp_time, &min_comp_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comp_time, &sum_comp_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_comp_time = sum_comp_time / numtasks;

        // Collect worker communication times
        MPI_Reduce(&worker_comm_time, &max_worker_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Convert matrix B back to its original form
        transpose(N, b);

        // Synchronize before ending timing
        MPI_Barrier(MPI_COMM_WORLD);
        stop = MPI_Wtime();

        // Calculate total time
        double total_time = stop - start;
        double total_comm_overhead = total_comm_time + max_worker_comm_time;

        printf("Matrix Size (N) = %d, Processors = %d\n", N, numworkers + 1);
        printf("\tTotal Time: %.6f s\n", total_time);
        printf("\tComputation Time (Max): %.6f s\n", max_comp_time);
        printf("\tComputation Time (Min): %.6f s\n", min_comp_time);
        printf("\tComputation Time (Avg): %.6f s\n", avg_comp_time);
        printf("\tCommunication Time (Master): %.6f s\n", total_comm_time);
        printf("\tCommunication Time (Workers Max): %.6f s\n", max_worker_comm_time);
        printf("\tLoad Imbalance: %.6f s (%.2f%%)\n",
               max_comp_time - min_comp_time,
               (max_comp_time > 0) ? ((max_comp_time - min_comp_time) / max_comp_time * 100.0) : 0.0);
        printf("\tParallel Efficiency: %.2f%%\n",
               (total_time > 0) ? (max_comp_time / total_time * 100.0) : 0.0);
        printf("\tCommunication Overhead: %.6f s (%.2f%%)\n\n",
               total_comm_overhead,
               (total_time > 0) ? (total_comm_overhead / total_time * 100.0) : 0.0);
    }

    /*---------------------------- WORKER ----------------------------*/
    if (taskid > 0) {
        source = 0;

        // Synchronize before starting timing
        MPI_Barrier(MPI_COMM_WORLD);

        // Measure communication time for receiving
        comm_start = MPI_Wtime();

        // Receive data from master
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, N * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

        comm_end = MPI_Wtime();
        double recv_time = comm_end - comm_start;

        // Measure computation time
        comp_start = MPI_Wtime();

        // Perform matrix multiplication
        for (i = 0; i < rows; i++) {
            for (j = 0; j < N; j++) {
                c[i][j] = 0;
                for (k = 0; k < N; k++) {
                    c[i][j] += a[i][k] * b[j][k];
                }
            }
        }

        comp_end = MPI_Wtime();
        comp_time = comp_end - comp_start;

        // Measure communication time for sending
        comm_start = MPI_Wtime();

        // Send results back to master
        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&c, rows * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

        comm_end = MPI_Wtime();
        double send_time = comm_end - comm_start;

        worker_comm_time = recv_time + send_time;

        // Participate in reduction
        MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comp_time, &min_comp_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comp_time, &sum_comp_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&worker_comm_time, &max_worker_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Synchronize before ending timing
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

/* Transpose a square matrix of given size */
void transpose(int size, double matrix[size][size]) {
    double temp;
    for (int i = 0; i < size - 1; i++) {
        for (int j = i; j < size; j++) {
            temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
}