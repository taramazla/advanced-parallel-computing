// incrementArray.cu
#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define ll long long

// Macro cek error CUDA biar kelihatan jelas kalau ada yang gagal
#define CHECK(call)                                                   \
    do {                                                              \
        cudaError_t e = (call);                                       \
        if (e != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(e));       \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Versi host tetap sama
void incrementArrayOnHost(float *a, ll N) {
    for (ll i = 0; i < N; i++) {
        a[i] = a[i] + 1.f;
    }
}

// KERNEL: pakai grid-stride loop supaya bisa handle N "sebarang" (besar banget)
__global__ void incrementArrayOnDevice(float *a, ll N) {
    ll idx    = blockIdx.x * (ll)blockDim.x + threadIdx.x;
    ll stride = (ll)blockDim.x * (ll)gridDim.x;

    // Setiap thread melompati indeks dengan kelipatan "stride"
    for (ll i = idx; i < N; i += stride) {
        a[i] = a[i] + 1.f;
    }
}

int main(int argc, char **argv) {
    float *a_h, *b_h;   // host
    float *a_d;         // device
    ll N        = 10LL;
    ll blockSize = 256LL;   // 256 thread per block (aman di semua GPU modern)

    // ARGUMENT:
    //  ./a.out              -> N=10 (default)
    //  ./a.out N blockSize  -> custom
    //
    // Untuk menguji melebihi kapasitas GTX1080 & RTX3080:
    //  ./a.out 3000000000 256
    // (3e9 float ~ 12 GB > 8 GB & 10 GB)
    if (argc == 3) {
        N = atoll(argv[1]);
        blockSize = atoll(argv[2]);
    } else {
        printf("Custom? jalankan: %s [N] [blockSize]\n", argv[0]);
        printf("Default N = %lld, blockSize = %lld\n\n", N, blockSize);
    }

    size_t size = (size_t)N * sizeof(float);

    // Info device
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Device info ===\n");
    printf("Name          : %s\n", prop.name);
    printf("Global memory : %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("===================\n\n");

    printf("N = %lld elemen float\n", N);
    printf("Perlu memori (array a_d saja) ≈ %.2f GB\n",
           size / (1024.0 * 1024.0 * 1024.0));
    printf("blockSize = %lld\n\n", blockSize);

    // ALLOKASI HOST
    a_h = (float *)malloc(size);
    b_h = (float *)malloc(size);
    if (a_h == NULL || b_h == NULL) {
        fprintf(stderr,
                "malloc() di host gagal (kemungkinan N terlalu besar untuk RAM host)\n");
        return EXIT_FAILURE;
    }

    // Inisialisasi host
    for (ll i = 0; i < N; i++) {
        a_h[i] = (float)i;
    }

    // ALLOKASI DEVICE – INI YANG AKAN GAGAL KALAU N > kapasitas GPU
    cudaError_t err = cudaMalloc((void **)&a_d, size);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cudaMalloc gagal: %s\n"
                "Kemungkinan besar N melebihi kapasitas global memory GPU.\n",
                cudaGetErrorString(err));
        free(a_h);
        free(b_h);
        return EXIT_SUCCESS; // program selesai, ini memang eksperimen "melebihi kapasitas"
    }

    // copy host -> device
    CHECK(cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice));

    // hitung di host
    incrementArrayOnHost(a_h, N);

    // hitung di device
    // NOTE: karena kernel pakai grid-stride loop, kita tidak wajib
    // punya N == nBlocks*blockSize lagi. Cukup pilih grid "wajar".
    ll maxBlocks =  (ll)prop.multiProcessorCount * 32; // heuristik: 32 block per SM
    ll nBlocks   = (N + blockSize - 1) / blockSize;
    if (nBlocks > maxBlocks) nBlocks = maxBlocks;

    printf("nBlocks yang dipakai = %lld\n\n", nBlocks);

    incrementArrayOnDevice<<<(int)nBlocks, (int)blockSize>>>(a_d, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // copy device -> host
    CHECK(cudaMemcpy(b_h, a_d, size, cudaMemcpyDeviceToHost));

    // cek hasil
    printf("Check assert...\n");
    for (ll i = 0; i < N; i++) {
        assert(a_h[i] == b_h[i]);
    }
    printf("Done, hasil benar.\n");

    // cleanup
    free(a_h);
    free(b_h);
    cudaFree(a_d);

    return EXIT_SUCCESS;
}
