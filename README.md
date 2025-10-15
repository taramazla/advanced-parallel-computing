# advanced-parallel-computing

##  Cluster 8 Node Fasilkom UI
ssh user05@152.118.31.24 -p 22

## Multi-Core Fasilkom UI
ssh user05@152.118.31.61 -p 22


mkdir UTS-Tara
cd UTS-Tara

nano matrix_vector_256.c
nano run-mat-vec.sh

nano matrix_matrix_256.c
nano run-mat-mat.sh

nano results.sh


Matrix-Vector multiplication (row-wise distribution)
Matrix-Matrix multiplication (standard row distribution)
Matrix-Matrix multiplication dengan menggunakan Cannon's algorithm (2D block distribution)


user05@store-01:~/coba3$ mpicc deadlock.c -o deadlock.o
user05@store-01:~/coba3$ mpirun -np 4 ./deadlock.o