# Parallel Conjugate Gradient Method - Dense SPD Matrices

This directory contains multiple implementations of the Conjugate Gradient (CG) method for solving dense symmetric positive-definite (SPD) linear systems **Ax = b**. The implementations target different parallel computing platforms:

- **CPU Sequential**: Single-threaded reference implementation
- **MPI Cluster**: Distributed memory parallelism for clusters
- **OpenMP Multicore**: Shared memory parallelism for multicore CPUs
- **CUDA Custom**: GPU implementation with custom CUDA kernels
- **CUDA cuBLAS**: GPU implementation using cuBLAS library
- **CUDA cuSPARSE**: GPU implementation with sparse matrix conversion

## Purpose

This code is designed for **Assignment Topic 3** (Parallel Algorithms for Large Linear Systems on GPU) comparing:
- **GPU** acceleration (CUDA implementations)
- **MPI Cluster** distributed computing
- **Multicore** shared memory parallelism

All implementations solve the same problem to enable fair performance comparison.

---

## Building

### Prerequisites
- **GCC/Clang**: C compiler with C99 support
- **MPI**: OpenMPI, MPICH, or Intel MPI
- **CUDA Toolkit**: Version 11.0+ (for GPU versions)
- **OpenMP**: Usually bundled with GCC

### Compile All Implementations
```bash
make all
```

### Compile Individual Targets
```bash
make cg_cpu        # CPU sequential
make cg_mpi        # MPI distributed
make cg_openmp     # OpenMP multicore
make cg_cuda       # CUDA custom kernels
make cg_cublas     # CUDA cuBLAS
make cg_cusparse   # CUDA cuSPARSE
```

### Clean Build
```bash
make clean
```

---

## Running

### CPU Sequential
```bash
./cg_cpu [matrix_size]
```
Example:
```bash
./cg_cpu 2048
```

### MPI Distributed
```bash
mpirun -np [num_processes] ./cg_mpi [matrix_size]
```
Example (4 processes):
```bash
mpirun -np 4 ./cg_mpi 2048
```

### OpenMP Multicore
```bash
export OMP_NUM_THREADS=[num_threads]
./cg_openmp [matrix_size]
```
Example (8 threads):
```bash
export OMP_NUM_THREADS=8
./cg_openmp 2048
```

### CUDA Versions
```bash
./cg_cuda [matrix_size]
./cg_cublas [matrix_size]
./cg_cusparse [matrix_size]
```
Example:
```bash
./cg_cuda 4096
```

---

## Algorithm Details

### Conjugate Gradient Method
The CG algorithm solves **Ax = b** where **A** is symmetric positive-definite:

```
1. Initialize: r₀ = b - Ax₀ (assuming x₀ = 0, so r₀ = b)
2. Set p₀ = r₀
3. For k = 0, 1, 2, ... until convergence:
   a. αₖ = (rₖᵀrₖ) / (pₖᵀApₖ)
   b. xₖ₊₁ = xₖ + αₖpₖ
   c. rₖ₊₁ = rₖ - αₖApₖ
   d. Check convergence: ||rₖ₊₁|| < tol × ||r₀||
   e. βₖ = (rₖ₊₁ᵀrₖ₊₁) / (rₖᵀrₖ)
   f. pₖ₊₁ = rₖ₊₁ + βₖpₖ
```

### Matrix Generation
All implementations use the same SPD matrix generation:
1. Generate random matrix **M**
2. Symmetrize: **A = (M + Mᵀ) / 2**
3. Make diagonally dominant: **A[i,i] = Σⱼ≠ᵢ |A[i,j]| + 1**

This ensures **A** is symmetric positive-definite with the same eigenvalues across all implementations (random seed = 42).

### Convergence Criteria
- **Relative tolerance**: `||rₖ|| < 1e-6 × ||r₀||`
- **Breakdown detection**: Stop if `|pᵀAp| < 1e-30` (prevents division by zero)
- **Maximum iterations**: 10,000

---

## Implementation Details

### CPU Sequential (`cg_cpu`)
- Pure C implementation
- Reference for correctness
- No parallelization
- Time: O(iterations × N²)

### MPI Distributed (`cg_mpi`)
- **Row-wise matrix partitioning**
- Each process stores `local_rows × N` matrix rows
- **Global operations**:
  - `MPI_Allreduce` for dot products (SUM)
  - `MPI_Allgatherv` for vector synchronization
- **Communication pattern**:
  - Allgatherv before each matvec (gather p_global)
  - Allreduce for r·r and p·Ap
- Scales with number of processes

### OpenMP Multicore (`cg_openmp`)
- **Shared memory parallelism**
- `#pragma omp parallel for` on:
  - Matrix-vector multiplication
  - Vector operations (axpy, scale)
- `reduction(+:sum)` for dot products
- Set threads: `export OMP_NUM_THREADS=N`

### CUDA Custom (`cg_cuda`)
- **Custom CUDA kernels**:
  - `matrix_vector_multiply`: Dense GEMV
  - `vector_add`, `vector_subtract`, `vector_scale`: Element-wise ops
- **Strided kernels**: Handle arbitrary matrix sizes beyond grid limits
- Uses cuBLAS for dot products (`cublasDdot`)
- Complete error checking with `CHECK_CUDA` macro

### CUDA cuBLAS (`cg_cublas`)
- **cuBLAS library functions**:
  - `cublasDgemv`: Matrix-vector multiplication
  - `cublasDdot`: Dot product
  - `cublasDaxpy`: AXPY (y = αx + y)
  - `cublasDscal`: Scaling (x = αx)
- Optimized library routines
- Less custom code, potentially better performance

### CUDA cuSPARSE (`cg_cusparse`)
- Converts dense matrix to CSR (Compressed Sparse Row) format
- Custom SpMV kernel (not using cuSPARSE API in current version)
- Useful for sparse matrices (many zero elements)

---

## Performance Considerations

### Memory Requirements
- **Dense matrix**: N² doubles = 8N² bytes
  - N=1024: ~8 MB
  - N=2048: ~33 MB
  - N=4096: ~134 MB
  - N=8192: ~536 MB
  - N=16384: ~2.1 GB

### Computational Complexity
- **Per iteration**:
  - Matvec (Ap): O(N²) FLOPs
  - Dot products: O(N) FLOPs
  - Vector updates: O(N) FLOPs
- **Total**: O(iterations × N²)

### Expected Performance Trends
1. **CPU Sequential**: Baseline, slowest for large N
2. **OpenMP**: ~linear speedup up to core count (memory bandwidth bound)
3. **MPI Cluster**: Speedup limited by communication overhead (Allgatherv dominant)
4. **CUDA**: Significant speedup for N > 2048 (large enough to saturate GPU)
5. **cuBLAS vs Custom**: cuBLAS often faster due to optimized kernels

### Scalability Tips
- **GPU**: Increase N for better GPU utilization (N ≥ 4096 recommended)
- **MPI**: Use fewer processes for small N to reduce communication overhead
- **OpenMP**: Thread count ≤ physical cores for best performance

---

## Experiment Guidelines

### Recommended Test Sizes
- Small: N = 1024, 2048
- Medium: N = 4096, 8192
- Large: N = 16384 (requires ~2GB memory)

### Processor Counts
- **OpenMP/MPI**: 1, 2, 4, 8, 16 threads/processes
- **GPU**: Single GPU (compare different CUDA implementations)

### Metrics to Collect
1. **Total execution time** (seconds)
2. **Iterations to convergence**
3. **Final residual** (verify correctness)
4. **Speedup** = Time(sequential) / Time(parallel)
5. **Efficiency** = Speedup / Num_processors

### Sample Experiment Script
```bash
#!/bin/bash
for N in 1024 2048 4096 8192; do
    echo "=== N=$N ==="

    # CPU Sequential
    ./cg_cpu $N

    # OpenMP (vary threads)
    for T in 1 2 4 8; do
        export OMP_NUM_THREADS=$T
        ./cg_openmp $N
    done

    # MPI (vary processes)
    for P in 1 2 4 8; do
        mpirun -np $P ./cg_mpi $N
    done

    # GPU versions
    ./cg_cuda $N
    ./cg_cublas $N
done
```

---

## Expected Output

### Example (CPU Sequential, N=2048)
```
CPU Sequential Conjugate Gradient
Matrix size: 2048 x 2048
Initial residual: 2.123456e+01
Converged after 1234 iterations
Iterations: 1234
Final residual: 1.234567e-06
Verification residual: 1.345678e-06
Total time: 12.345678 seconds
```

### MPI Output (4 processes)
```
MPI Parallel Conjugate Gradient
Processes: 4
Matrix size: 2048 x 2048
Initial residual: 2.123456e+01
Converged after 1234 iterations
Iterations: 1234
Final residual: 1.234567e-06
Verification residual: 1.345678e-06
Total time: 3.456789 seconds
```

---

## Troubleshooting

### Build Errors
- **CUDA not found**: Install CUDA Toolkit, set `PATH` and `LD_LIBRARY_PATH`
- **MPI not found**: Install OpenMPI or MPICH
- **OpenMP not supported**: Use GCC 4.9+ or Clang 3.7+

### Runtime Errors
- **Out of memory**: Reduce N or use fewer MPI processes
- **GPU errors**: Check CUDA driver version matches toolkit
- **MPI hangs**: Check network configuration, firewall settings

### Performance Issues
- **Slow GPU**: N too small (increase to 4096+)
- **MPI no speedup**: Communication overhead dominates (increase N or reduce processes)
- **OpenMP no speedup**: Memory bandwidth bound (typical for dense matvec)

---

## References

1. **Textbook**: *Parallel Programming for Multicore and Cluster Systems* (Rauber & Rünger), Chapter 7
2. **CG Algorithm**: Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients for solving linear systems.
3. **CUDA Programming**: NVIDIA CUDA C Programming Guide
4. **MPI Standard**: Message Passing Interface Forum

---

## Assignment Deliverables

For the assignment, provide:

1. **Experiment Results**: CSV/table with N, processors, time, speedup, efficiency
2. **Performance Plots**:
   - Speedup vs. Number of Processors (for fixed N)
   - Execution Time vs. Problem Size (for different implementations)
3. **Analysis**:
   - Compare GPU vs. MPI vs. Multicore
   - Discuss scalability and bottlenecks
   - Explain communication overhead in MPI
   - Analyze GPU performance characteristics
4. **Code Submission**: All source files + Makefile + README

---

## Contact

For questions or issues, refer to:
- Course materials (Chapter 7)
- CUDA documentation: https://docs.nvidia.com/cuda/
- MPI tutorials: https://mpitutorial.com/

**Good luck with your experiments!**
