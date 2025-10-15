#!/bin/bash

# Check if results directory exists
if [ ! -d "results" ]; then
    echo "Results directory not found. Have the jobs completed?"
    exit 1
fi

echo "==============================================================="
echo "                  Results          "
echo "==============================================================="
echo "Matrix Size | Processors | Total Time | Execution Time | Communication Time"
echo "-----------+------------+------------+---------------+------------------"

# Parse all output files
for size in 256 512 1024 2048 4096
do
    for procs in 1 2 4 8 16 32
    do
        result_file="results/run-N${size}-P${procs}.out"

        if [ -f "$result_file" ]; then
            # Extract timing information using grep and awk
            total_time=$(grep "Max Total Execution Time:" "$result_file" | awk '{print $5}')
            exec_time=$(grep "Max Computation Time:" "$result_file" | awk '{print $4}')
            comm_time=$(grep "Avg Communication Time:" "$result_file" | awk '{print $4}')

            printf "%-11s | %-10s | %-10s | %-13s | %-16s\n" "${size}" "${procs}" "${total_time}" "${exec_time}" "${comm_time}"
        else
            printf "%-11s | %-10s | %-10s | %-13s | %-16s\n" "${size}" "${procs}" "N/A" "N/A" "N/A"
        fi
    done
    echo "-----------+------------+------------+---------------+------------------"
done

echo "==============================================================="