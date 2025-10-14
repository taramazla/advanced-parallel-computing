#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <stdint.h>
#include <stdlib.h>

#define N 256  /* number of rows and columns in matrix */

double a[N][N], b[N], c[N];
MPI_Status status;

int main(int argc, char **argv) {
    int numtasks, taskid, numworkers, rows, offset;
    double start, end, comm_start, comm_end, recv_start, recv_end;
    double comp_start, comp_end, comp_time = 0.0;
    double max_comp_time, min_comp_time, avg_comp_time, sum_comp_time;
    double worker_comm_time = 0.0, max_worker_comm_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    numworkers = numtasks - 1;

    /*---------------------------- master process ----------------------------*/
    if (taskid == 0) {
        srand(time(NULL));

        // Initialize matrix 'a' and vector 'b' with random values
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = (double)rand() / RAND_MAX;
            }
        }

        for (int i = 0; i < N; i++) {
            b[i] = (double)rand() / RAND_MAX;
        }

        if (numworkers > 0) {
            rows = N / numworkers;
        } else {
            rows = N;
        }

        // Synchronize all processes before starting timing
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();

        offset = 0;
        comm_start = MPI_Wtime();

        // Send data to workers
        int send_rows = rows;
        for (int dest = 1; dest <= numworkers; dest++) {
            if (dest == numworkers) {
                send_rows = N - offset;
            }

            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&send_rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(a[offset], send_rows * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            MPI_Send(b, N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);

            offset = offset + rows;
        }

        comm_end = MPI_Wtime();

        recv_start = MPI_Wtime();
        // Receive results from workers
        for (int source = 1; source <= numworkers; source++) {
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset], rows, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }
        recv_end = MPI_Wtime();

        /*--------------- Sequential computation if only 1 processor ---------------*/
        if (numworkers == 0) {
            comp_start = MPI_Wtime();
            for (int i = 0; i < N; i++) {
                c[i] = 0;
                for (int j = 0; j < N; j++) {
                    c[i] += a[i][j] * b[j];
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

        // Synchronize before ending timing
        MPI_Barrier(MPI_COMM_WORLD);
        end = MPI_Wtime();

        // Calculate timing metrics
        double total_time = end - start;
        double master_comm_time = (comm_end - comm_start) + (recv_end - recv_start);
        double total_comm_overhead = master_comm_time + max_worker_comm_time;

        // Output timing results
        printf("Matrix Size (N) = %d, Processor = %d\n", N, numworkers + 1);
        printf("Tara Mazaya Lababan - 2406514564\n");
        printf("Total Time: %.6f s\n", total_time);
        printf("Computation Time (Max): %.6f s\n", max_comp_time);
        printf("Computation Time (Min): %.6f s\n", min_comp_time);
        printf("Computation Time (Avg): %.6f s\n", avg_comp_time);
        printf("Communication Time (Master): %.6f s\n", master_comm_time);
        printf("Communication Time (Workers Max): %.6f s\n", max_worker_comm_time);
        printf("Load Imbalance: %.6f s (%.2f%%)\n",
               max_comp_time - min_comp_time,
               (max_comp_time > 0) ? ((max_comp_time - min_comp_time) / max_comp_time * 100.0) : 0.0);
        printf("Parallel Efficiency: %.2f%%\n",
               (total_time > 0) ? (max_comp_time / total_time * 100.0) : 0.0);
        printf("Communication Overhead: %.6f s (%.2f%%)\n\n",
               total_comm_overhead,
               (total_time > 0) ? (total_comm_overhead / total_time * 100.0) : 0.0);
    }

    /*---------------------------- worker process ----------------------------*/
    if (taskid > 0) {
        int source = 0;

        // Synchronize before starting timing
        MPI_Barrier(MPI_COMM_WORLD);

        // Measure communication time for receiving
        comm_start = MPI_Wtime();

        // Receive data from master
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(a, rows * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(b, N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

        comm_end = MPI_Wtime();
        double recv_time = comm_end - comm_start;

        // Measure computation time
        comp_start = MPI_Wtime();

        // Perform matrix-vector multiplication
        for (int i = 0; i < rows; i++) {
            c[i] = 0;
            for (int j = 0; j < N; j++) {
                c[i] += a[i][j] * b[j];
            }
        }

        comp_end = MPI_Wtime();
        comp_time = comp_end - comp_start;

        // Measure communication time for sending
        comm_start = MPI_Wtime();

        // Send results back to master
        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(c, rows, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

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