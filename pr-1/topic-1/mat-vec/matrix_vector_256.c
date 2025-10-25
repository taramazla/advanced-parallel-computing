#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <stdint.h>
#include <stdlib.h>

#define N 256  /* number of rows and columns in matrix */

double a[N][N], b[N], c[N];
MPI_Status status;

int main(int argc, char **argv) {
    int numtasks, taskid, rows, offset, my_offset, my_rows;
    double start, end, comm_start, comm_end;
    double comp_start, comp_end, comp_time = 0.0;
    double max_comp_time, min_comp_time, avg_comp_time, sum_comp_time;
    double worker_comm_time = 0.0, max_worker_comm_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // Initialize data on rank 0
    if (taskid == 0) {
        srand(12345); // Use fixed seed for consistent results

        // Initialize matrix 'a' and vector 'b' with random values
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = (double)rand() / RAND_MAX;
            }
        }

        for (int i = 0; i < N; i++) {
            b[i] = (double)rand() / RAND_MAX;
        }
    }

    // Calculate work distribution
    rows = N / numtasks;
    int remaining = N % numtasks;
    
    // Calculate this process's work assignment
    my_rows = rows;
    if (taskid < remaining) {
        my_rows++;
        my_offset = taskid * my_rows;
    } else {
        my_offset = remaining * (rows + 1) + (taskid - remaining) * rows;
    }

    // Synchronize all processes before starting timing
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    // Broadcast matrix and vector data to all processes
    comm_start = MPI_Wtime();
    MPI_Bcast(a, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    comm_end = MPI_Wtime();
    
    if (taskid == 0) {
        worker_comm_time += (comm_end - comm_start);
    } else {
        worker_comm_time += (comm_end - comm_start);
    }

    // All processes perform computation on their assigned rows
    comp_start = MPI_Wtime();

    for (int i = 0; i < my_rows; i++) {
        c[i] = 0;
        for (int j = 0; j < N; j++) {
            c[i] += a[my_offset + i][j] * b[j];
        }
    }

    comp_end = MPI_Wtime();
    comp_time = comp_end - comp_start;

    // Gather results from all processes
    if (taskid == 0) {
        comm_start = MPI_Wtime();
        
        // Receive results from other processes
        for (int source = 1; source < numtasks; source++) {
            int src_rows = rows;
            int src_offset;
            
            if (source < remaining) {
                src_rows++;
                src_offset = source * src_rows;
            } else {
                src_offset = remaining * (rows + 1) + (source - remaining) * rows;
            }
            
            MPI_Recv(&c[src_offset], src_rows, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }
        
        comm_end = MPI_Wtime();
        worker_comm_time += (comm_end - comm_start);
    } else {
        comm_start = MPI_Wtime();
        
        MPI_Send(c, my_rows, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        
        comm_end = MPI_Wtime();
        worker_comm_time += (comm_end - comm_start);
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

    // Output results from rank 0
    if (taskid == 0) {
        double total_time = end - start;
        double total_comm_overhead = max_worker_comm_time;

        printf("Matrix Size (N) = %d, Processor = %d\n", N, numtasks);
        printf("Tara Mazaya Lababan - 2406514564\n");
        printf("Total Time: %.6f s\n", total_time);
        printf("Computation Time (Max): %.6f s\n", max_comp_time);
        printf("Computation Time (Min): %.6f s\n", min_comp_time);
        printf("Computation Time (Avg): %.6f s\n", avg_comp_time);
        printf("Communication Time (Max): %.6f s\n", max_worker_comm_time);
        printf("Load Imbalance: %.6f s (%.2f%%)\n",
               max_comp_time - min_comp_time,
               (max_comp_time > 0) ? ((max_comp_time - min_comp_time) / max_comp_time * 100.0) : 0.0);
        printf("Parallel Efficiency: %.2f%%\n",
               (total_time > 0) ? (max_comp_time / total_time * 100.0) : 0.0);
        printf("Communication Overhead: %.6f s (%.2f%%)\n\n",
               total_comm_overhead,
               (total_time > 0) ? (total_comm_overhead / total_time * 100.0) : 0.0);
        printf("-------------------------------------------------------\n");
        printf("\n");
        printf("Result Matrix C:\n");
        for (int i = 0; i < N; i++) {
            printf("c[%d] = %.6f\n", i, c[i]);
        }
    }

    MPI_Finalize();

    return 0;
}