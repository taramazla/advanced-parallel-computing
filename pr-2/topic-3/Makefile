# Compiler settings
CC = gcc
MPICC = mpicc
NVCC = nvcc

# Compiler flags
CFLAGS = -O3 -Wall
OMPFLAGS = -fopenmp
NVCCFLAGS = -O3 -arch=sm_60
CUBLASFLAGS = -lcublas

# Targets
TARGETS = cg_cpu cg_mpi cg_openmp cg_cuda cg_cublas cg_cusparse

.PHONY: all clean

all: $(TARGETS)

# CPU Sequential Version
cg_cpu: conjugate_gradient_cpu.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# MPI Distributed Version
cg_mpi: conjugate_gradient_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $< -lm

# OpenMP Multicore Version
cg_openmp: conjugate_gradient_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< -lm

# CUDA Custom Kernels Version
cg_cuda: conjugate_gradient_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(CUBLASFLAGS)

# CUDA cuBLAS Version
cg_cublas: conjugate_gradient_cublas.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(CUBLASFLAGS)

# CUDA cuSPARSE Version
cg_cusparse: conjugate_gradient_cusparse.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(CUBLASFLAGS) -lcusparse

clean:
	rm -f $(TARGETS) *.o
