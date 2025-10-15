#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Comm group_comm;
    MPI_Group world_group, new_group;

    // Inisialisasi MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Dapatkan grup dari MPI_COMM_WORLD
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // Buat dua array untuk rank genap dan ganjil
    int ranks_even[size/2], ranks_odd[size/2];
    int even_count = 0, odd_count = 0;
    for (int i = 0; i < size; i++) {
        if (i % 2 == 0) {
            ranks_even[even_count++] = i; // Rank genap
        } else {
            ranks_odd[odd_count++] = i;   // Rank ganjil
        }
    }

    // Tentukan grup berdasarkan rank proses
    if (rank % 2 == 0) {
        // Proses dengan rank genap masuk ke grup genap
        MPI_Group_incl(world_group, even_count, ranks_even, &new_group);
    } else {
        // Proses dengan rank ganjil masuk ke grup ganjil
        MPI_Group_incl(world_group, odd_count, ranks_odd, &new_group);
    }

    // Buat communicator baru untuk grup
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &group_comm);

    // Verifikasi grup dan communicator
    if (group_comm != MPI_COMM_NULL) {
        int new_rank, new_size;
        MPI_Comm_rank(group_comm, &new_rank);
        MPI_Comm_size(group_comm, &new_size);
        printf("Rank global %d menjadi rank %d di grup %s (ukuran: %d)\n",
               rank, new_rank, (rank % 2 == 0) ? "genap" : "ganjil", new_size);
        fflush(stdout);

        // Komunikasi sederhana dalam grup: broadcast dari rank 0 di grup
        int data = 0;
        if (new_rank == 0) {
            data = rank * 100; // Nilai berdasarkan rank global
            printf("Proses %d (rank %d di grup) mengirim data %d\n", rank, new_rank, data);
            fflush(stdout);
        }
        MPI_Bcast(&data, 1, MPI_INT, 0, group_comm);
        printf("Proses %d (rank %d di grup) menerima data %d\n", rank, new_rank, data);
        fflush(stdout);
    } else {
        printf("Proses %d tidak termasuk dalam grup\n", rank);
        fflush(stdout);
    }

    if (group_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&group_comm);
    }
    MPI_Group_free(&new_group);
    MPI_Group_free(&world_group);

    MPI_Finalize();
    return 0;
}