#!/bin/bash

# Check if results directory exists
if [ ! -d "results" ]; then
    echo "Results directory not found. Have the jobs completed?"
    exit 1
fi

echo "======================================================================="
echo "                    Multiplication Results                            "
echo "======================================================================="
echo "Matrix Size | Processors | Total Time | Computation | Communication | Parallel  "
echo "            |            | (s)        | Time (s)    | Time (s)      | Eff. (%)  "
echo "------------+------------+------------+-------------+---------------+-----------"

# Parse all output files
for size in 256 512 1024 2048 4096
do
    for procs in 1 2 4 8 16 32
    do
        result_file="results/run-N${size}-P${procs}.out"

        if [ -f "$result_file" ]; then
            # Extract timing information using grep and awk
            total_time=$(grep "Total Time:" "$result_file" | awk '{print $3}')
            comp_time=$(grep "Computation Time (Max):" "$result_file" | awk '{print $4}')
            comm_time=$(grep "Communication Overhead:" "$result_file" | awk '{print $3}')
            par_eff=$(grep "Parallel Efficiency:" "$result_file" | awk '{print $3}' | tr -d '%')

            printf "%-11s | %-10s | %-10s | %-11s | %-13s | %-9s\n" \
                   "${size}" "${procs}" "${total_time}" "${comp_time}" "${comm_time}" "${par_eff}"
        else
            printf "%-11s | %-10s | %-10s | %-11s | %-13s | %-9s\n" \
                   "${size}" "${procs}" "N/A" "N/A" "N/A" "N/A"
        fi
    done
    echo "------------+------------+------------+-------------+---------------+-----------"
done

echo "======================================================================="