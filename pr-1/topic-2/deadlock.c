#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    int buf_size = 1000000; // Data besar untuk memaksa deadlock
    int *buf = malloc(buf_size * sizeof(int));
    int *result = malloc(buf_size * sizeof(int));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Inisialisasi buffer dengan data besar
    for (int i = 0; i < buf_size; i++) {
        buf[i] = rank + 1;
    }

    if (rank == 0) {
        printf("Proses %d memanggil Bcast...\n", rank);
        MPI_Reduce(buf, result, buf_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        printf("Proses %d memanggil Reduce...\n", rank);
        MPI_Bcast(buf, buf_size, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        printf("Proses %d memanggil Reduce...\n", rank);
        MPI_Reduce(buf, result, buf_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        printf("Proses %d memanggil Bcast...\n", rank);
        MPI_Bcast(buf, buf_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    // if (rank == 0) {
    //     printf("Proses %d memanggil Bcast...\n", rank);
    //     MPI_Bcast(buf, buf_size, MPI_INT, 0, MPI_COMM_WORLD);
    //     printf("Proses %d memanggil Reduce...\n", rank);
    //     MPI_Reduce(buf, result, buf_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // } else {
    //     printf("Proses %d memanggil Reduce...\n", rank);
    //     MPI_Reduce(buf, result, buf_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    //     printf("Proses %d memanggil Bcast...\n", rank);
    //     MPI_Bcast(buf, buf_size, MPI_INT, 0, MPI_COMM_WORLD);
    // }
    printf("Proses %d: Selesai - hasil di root[0] = %d\n", rank, result[0]);

    free(buf);
    free(result);

    MPI_Finalize();
    return 0;
}