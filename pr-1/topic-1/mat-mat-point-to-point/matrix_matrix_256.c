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
    int numtasks, taskid, source, dest, rows, offset, i, j, k;
    int my_offset, my_rows;

    double start, stop;
    double comm_start, comm_end;
    double send_comm_time = 0.0, recv_comm_time = 0.0, total_comm_time = 0.0;
    double comp_start, comp_end, comp_time = 0.0;
    double max_comp_time, min_comp_time, avg_comp_time, sum_comp_time;
    double worker_comm_time = 0.0, max_worker_comm_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    /*---------------------------- MASTER INITIALIZATION (taskid 0 only) ----------------------------*/
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

        // Calculate rows per worker (including master as worker)
        rows = N / numtasks;
        int remaining_rows = N % numtasks;

        // Synchronize all processes before starting timing
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();

        // Master's own work portion (first worker)
        my_rows = rows + (0 < remaining_rows ? 1 : 0);
        my_offset = 0;
        offset = my_rows;  // Start offset for other workers

        // Measure time for sending data to other workers
        comm_start = MPI_Wtime();

        // Send data to other worker processes
        for (dest = 1; dest < numtasks; dest++) {
            int worker_rows = rows + (dest < remaining_rows ? 1 : 0);
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&worker_rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], worker_rows * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&b, N * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            offset += worker_rows;
        }

        comm_end = MPI_Wtime();
        send_comm_time = comm_end - comm_start;
        total_comm_time += send_comm_time;
    }

    /*---------------------------- ALL PROCESSES AS WORKERS ----------------------------*/
    
    // Non-master processes receive their work
    if (taskid > 0) {
        // Synchronize before starting timing
        MPI_Barrier(MPI_COMM_WORLD);

        // Measure communication time for receiving
        comm_start = MPI_Wtime();

        // Receive data from master
        MPI_Recv(&my_offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&my_rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, my_rows * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, N * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);

        comm_end = MPI_Wtime();
        double recv_time = comm_end - comm_start;
        worker_comm_time += recv_time;
    }

    // All processes perform computation
    comp_start = MPI_Wtime();

    // Perform matrix multiplication
    for (i = 0; i < my_rows; i++) {
        for (j = 0; j < N; j++) {
            c[i][j] = 0;
            for (k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[j][k];
            }
        }
    }

    comp_end = MPI_Wtime();
    comp_time = comp_end - comp_start;

    // Non-master processes send results back
    if (taskid > 0) {
        // Measure communication time for sending
        comm_start = MPI_Wtime();

        // Send results back to master
        MPI_Send(&my_offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&my_rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&c, my_rows * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

        comm_end = MPI_Wtime();
        double send_time = comm_end - comm_start;
        worker_comm_time += send_time;
    }

    /*---------------------------- MASTER COLLECTION (taskid 0 only) ----------------------------*/
    if (taskid == 0) {
        // Master already has its own results in c[0] to c[my_rows-1]
        
        // Measure time for receiving results from other workers
        comm_start = MPI_Wtime();

        // Receive results from other worker processes
        for (i = 1; i < numtasks; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows * N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }

        comm_end = MPI_Wtime();
        recv_comm_time = comm_end - comm_start;
        total_comm_time += recv_comm_time;
    }

    // Collect computation times from all processes
    MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &min_comp_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &sum_comp_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_comp_time = sum_comp_time / numtasks;

    // Collect worker communication times
    MPI_Reduce(&worker_comm_time, &max_worker_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (taskid == 0) {
        // Convert matrix B back to its original form
        transpose(N, b);

        // Synchronize before ending timing
        MPI_Barrier(MPI_COMM_WORLD);
        stop = MPI_Wtime();

        // Calculate total time
        double total_time = stop - start;
        double total_comm_overhead = total_comm_time + max_worker_comm_time;

        printf("Matrix Size (N) = %d, Processors = %d\n", N, numtasks);
        printf("Tara Mazaya Lababan - 2406514564\n");
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

        printf("Result Matrix C:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%.2f ", c[i][j]);
            }
            printf("\n");
        }
    } else {
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