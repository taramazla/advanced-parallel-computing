#!/bin/bash

# Compile all matrix sizes
for size in 256 512 1024 2048 4096
do
    echo "Compiling matrix_vector_${size}.c"
    # First copy the base code
    cp matrix_vector_256.c matrix_vector_${size}.c

    # Replace the matrix size in the copied file
    sed -i.bak "s/#define N 256/#define N ${size}/" matrix_vector_${size}.c
    rm -f matrix_vector_${size}.c.bak

    # Compile with MPI
    mpicc -o matrix_vector_${size}.o matrix_vector_${size}.c

    echo "Compiled matrix_vector_${size}.o"
    echo "-----------------------------------"
done

# Create output directory for results
mkdir -p results

# Run all combinations
for size in 256 512 1024 2048 4096
do
    for procs in 1 2 4 8 16 32
    do
        # Calculate required nodes (8 processors per node)
        nodes=$(( (procs + 7) / 8 ))

        echo "Running with matrix size ${size} on ${procs} processors (${nodes} nodes)"

        # Create a job script for this combination
        cat > job_${size}_${procs}.sh << EOF
mpirun --hostfile hostfile -np ${procs} \$HOME/kelompok2/topik1/perkalian_matrix_vector/matrix_vector_${size}.o
EOF

        # Submit the job
        echo "Submitting job: sbatch job_${size}_${procs}.sh"

        # run the job script directly
        bash job_${size}_${procs}.sh > results/run-N${size}-P${procs}.out

        echo "-----------------------------------"

        # Small delay between job submissions to prevent overloading the scheduler
        sleep 1
    done
done

echo "All jobs submitted. Results will be in the 'results' directory."