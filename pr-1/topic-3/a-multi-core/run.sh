#!/bin/bash

# Compile all matrix sizes
for size in 256 512 1024 2048 4096
do
    echo "Compiling cgm${size}.c"
    # First copy the base code
    cp cgm256.c cgm${size}.c

    # Replace the matrix size in the copied file
    sed -i.bak "s/#define N 256/#define N ${size}/" cgm${size}.c
    rm -f cgm${size}.c.bak

    # Compile with MPI
    mpicc -o cgm${size}.o cgm${size}.c -lm

    echo "Compiled cgm${size}.o"
    echo "-----------------------------------"
done

# Create output directory for results
mkdir -p results

# Run all combinations
for size in 256 512 1024 2048 4096
do
    for procs in 1 2 4 8 16 32
    do
        # Calculate required nodes (32 processors per node)
        nodes=$(( (procs + 31) / 32 ))

        echo "Running with matrix size ${size} on ${procs} processors (${nodes} nodes)"

        # Create a job script for this combination
        cat > job_${size}_${procs}.sh << EOF
mpirun -np ${procs} ./cgm${size}.o
EOF

        # Submit the job
        echo "Running job: bash job_${size}_${procs}.sh"

        # run the job script directly
        bash job_${size}_${procs}.sh > results/run-N${size}-P${procs}.out

        echo "-----------------------------------"

        # Small delay between job submissions to prevent overloading the scheduler
        sleep 1
    done
done

echo "All jobs completed. Results are in the 'results' directory."