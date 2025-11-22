#include <stdio.h>
#include <stdlib.h>

__global__ void kernel1(int *a, int size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < size)
      a[idx] = 7;
}

__global__ void kernel2(int *a, int size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < size)
      a[idx] = blockIdx.x;
}

__global__ void kernel3(int *a, int size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < size)
      a[idx] = threadIdx.x;
}

int main(int argc, char **argv) {
    int size = 15;
    int threadsInBlock = 5;
    int blocksInGrid = 3;

    int *a1_d, *a2_d, *a3_d;
    int *a1_h, *a2_h, *a3_h;

    if(argc == 3) {
      threadsInBlock = atoi(argv[1]);
      blocksInGrid = atoi(argv[2]);
      size = threadsInBlock * blocksInGrid;
    } else {
      printf("Custom block size? pass [GRID_SIZE] [BLOCK_SIZE] as arguments\n\n");
    }


    a1_h = (int *) malloc (size * sizeof(int));
    a2_h = (int *) malloc (size * sizeof(int));
    a3_h = (int *) malloc (size * sizeof(int));

    cudaMalloc((void **) &a1_d, size * sizeof(int));
    cudaMalloc((void **) &a2_d, size * sizeof(int));
    cudaMalloc((void **) &a3_d, size * sizeof(int));

    kernel1<<<blocksInGrid, threadsInBlock>>>(a1_d, size);
    kernel2<<<blocksInGrid, threadsInBlock>>>(a2_d, size);
    kernel3<<<blocksInGrid, threadsInBlock>>>(a3_d, size);

    cudaMemcpy(a1_h, a1_d, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(a2_h, a2_d, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(a3_h, a3_d, size * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; i++) {
      printf("%d ", a1_h[i]);
    }
    printf("\n\n");

    for(int i = 0; i < size; i++) {
      printf("%d ", a2_h[i]);
    }
    printf("\n\n");

    for(int i = 0; i < size; i++) {
      printf("%d ", a3_h[i]);
    }
    printf("\n\n");

    free(a1_h); free(a2_h); free(a3_h);
    cudaFree(a1_d); cudaFree(a2_d); cudaFree(a3_d);
    return 0;
}