#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Comm cart_comm;
    int dims[2], periods[2], coords[2];
    int left, right, up, down;

    // Inisialisasi MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Tentukan dimensi grid 2D
    dims[0] = (int)sqrt(size); // Baris
    dims[1] = size / dims[0];  // Kolom

    // Jika tidak bisa membentuk grid persegi sempurna, adjust
    while (dims[0] * dims[1] != size) {
        dims[0]--;
        dims[1] = size / dims[0];
    }

    // Set periodic boundaries (wraparound)
    periods[0] = 1; // Periodic di dimensi baris
    periods[1] = 1; // Periodic di dimensi kolom

    printf("Rank %d: Membuat topologi Cartesian %dx%d\n", rank, dims[0], dims[1]);
    fflush(stdout);

    // Buat topologi Cartesian
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    if (cart_comm != MPI_COMM_NULL) {
        // Dapatkan koordinat proses dalam grid
        MPI_Cart_coords(cart_comm, rank, 2, coords);

        // Dapatkan tetangga dalam 4 arah
        MPI_Cart_shift(cart_comm, 0, 1, &up, &down);    // Tetangga vertikal
        MPI_Cart_shift(cart_comm, 1, 1, &left, &right); // Tetangga horizontal

        printf("Rank %d: Koordinat (%d,%d), Tetangga - Atas:%d, Bawah:%d, Kiri:%d, Kanan:%d\n",
               rank, coords[0], coords[1], up, down, left, right);
        fflush(stdout);

        // Komunikasi sederhana: kirim data ke tetangga kanan
        int send_data = rank;
        int recv_data = -1;

        MPI_Status status;

        // Kirim ke tetangga kanan, terima dari tetangga kiri
        MPI_Sendrecv(&send_data, 1, MPI_INT, right, 0,
                     &recv_data, 1, MPI_INT, left, 0,
                     cart_comm, &status);

        printf("Rank %d mengirim %d ke rank %d dan menerima %d dari rank %d\n",
               rank, send_data, right, recv_data, left);
        fflush(stdout);

        // Contoh penggunaan MPI_Cart_rank untuk konversi koordinat ke rank
        int test_coords[2] = {0, 0};
        int corner_rank;
        MPI_Cart_rank(cart_comm, test_coords, &corner_rank);

        if (rank == 0) {
            printf("Koordinat (0,0) memiliki rank: %d\n", corner_rank);
            fflush(stdout);
        }

        MPI_Comm_free(&cart_comm);
    } else {
        printf("Rank %d: Gagal membuat topologi Cartesian\n", rank);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}