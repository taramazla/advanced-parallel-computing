#!/bin/bash

# Experiment automation script for Conjugate Gradient implementations
# Compares GPU (CUDA) vs MPI (Cluster) vs Multicore (OpenMP) performance

# Output directory
OUTPUT_DIR="experiment_results"
mkdir -p "$OUTPUT_DIR"

# Results CSV file
RESULTS_CSV="$OUTPUT_DIR/cg_results.csv"

# Problem sizes to test
SIZES=(1024 2048 4096 8192)

# Processor counts for MPI and OpenMP
PROCS=(1 2 4 8)

# Number of runs per configuration (for averaging)
NUM_RUNS=3

# Initialize CSV with header
echo "Implementation,N,Processors,Run,Iterations,FinalResidual,VerificationResidual,TotalTime" > "$RESULTS_CSV"

# Function to extract metrics from output
extract_metrics() {
    local output="$1"
    local iterations=$(echo "$output" | grep "Iterations:" | awk '{print $2}')
    local final_res=$(echo "$output" | grep "Final residual:" | awk '{print $3}')
    local verify_res=$(echo "$output" | grep "Verification residual:" | awk '{print $3}')
    local total_time=$(echo "$output" | grep "Total time:" | awk '{print $3}')

    # Handle empty values
    iterations=${iterations:-0}
    final_res=${final_res:-0}
    verify_res=${verify_res:-0}
    total_time=${total_time:-0}

    echo "$iterations,$final_res,$verify_res,$total_time"
}

# Function to log result to CSV
log_result() {
    local impl="$1"
    local N="$2"
    local procs="$3"
    local run="$4"
    local metrics="$5"

    echo "$impl,$N,$procs,$run,$metrics" >> "$RESULTS_CSV"
}

echo "=========================================="
echo "CG Performance Experiment"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Results file: $RESULTS_CSV"
echo ""

# ==========================================
# CPU Sequential Baseline
# ==========================================
echo "[1/6] Running CPU Sequential..."
for N in "${SIZES[@]}"; do
    echo "  N=$N"
    for run in $(seq 1 $NUM_RUNS); do
        output=$(./cg_cpu $N 2>&1)
        if [ $? -ne 0 ]; then
            echo "    Run $run: ERROR - Execution failed"
            log_result "CPU_Sequential" "$N" "1" "$run" "0,0,0,0"
        else
            metrics=$(extract_metrics "$output")
            log_result "CPU_Sequential" "$N" "1" "$run" "$metrics"
            time_val=$(echo "$metrics" | awk -F, '{print $4}')
            echo "    Run $run: ${time_val} sec"
        fi
    done
done
echo ""

# ==========================================
# OpenMP Multicore
# ==========================================
echo "[2/6] Running OpenMP Multicore..."
for N in "${SIZES[@]}"; do
    echo "  N=$N"
    for P in "${PROCS[@]}"; do
        echo "    Threads=$P"
        for run in $(seq 1 $NUM_RUNS); do
            export OMP_NUM_THREADS=$P
            output=$(./cg_openmp $N 2>&1)
            if [ $? -ne 0 ]; then
                echo "      Run $run: ERROR - Execution failed"
                log_result "OpenMP_Multicore" "$N" "$P" "$run" "0,0,0,0"
            else
                metrics=$(extract_metrics "$output")
                log_result "OpenMP_Multicore" "$N" "$P" "$run" "$metrics"
                time_val=$(echo "$metrics" | awk -F, '{print $4}')
                echo "      Run $run: ${time_val} sec"
            fi
        done
    done
done
echo ""

# ==========================================
# MPI Cluster
# ==========================================
echo "[3/6] Running MPI Cluster..."
for N in "${SIZES[@]}"; do
    echo "  N=$N"
    for P in "${PROCS[@]}"; do
        echo "    Processes=$P"
        for run in $(seq 1 $NUM_RUNS); do
            output=$(mpirun --allow-run-as-root -np $P ./cg_mpi $N 2>&1)
            if [ $? -ne 0 ]; then
                echo "      Run $run: ERROR - MPI execution failed"
                echo "      Output: $output"
                log_result "MPI_Cluster" "$N" "$P" "$run" "0,0,0,0"
            else
                metrics=$(extract_metrics "$output")
                log_result "MPI_Cluster" "$N" "$P" "$run" "$metrics"
                time_val=$(echo "$metrics" | awk -F, '{print $4}')
                echo "      Run $run: ${time_val} sec"
            fi
        done
    done
done
echo ""

# ==========================================
# CUDA Custom Kernels
# ==========================================
echo "[4/6] Running CUDA Custom Kernels..."
for N in "${SIZES[@]}"; do
    echo "  N=$N"
    for run in $(seq 1 $NUM_RUNS); do
        output=$(./cg_cuda $N 2>&1)
        if [ $? -ne 0 ]; then
            echo "    Run $run: ERROR - CUDA execution failed"
            log_result "CUDA_Custom" "$N" "1" "$run" "0,0,0,0"
        else
            metrics=$(extract_metrics "$output")
            log_result "CUDA_Custom" "$N" "1" "$run" "$metrics"
            time_val=$(echo "$metrics" | awk -F, '{print $4}')
            echo "    Run $run: ${time_val} sec"
        fi
    done
done
echo ""

# ==========================================
# CUDA cuBLAS
# ==========================================
echo "[5/6] Running CUDA cuBLAS..."
for N in "${SIZES[@]}"; do
    echo "  N=$N"
    for run in $(seq 1 $NUM_RUNS); do
        output=$(./cg_cublas $N 2>&1)
        if [ $? -ne 0 ]; then
            echo "    Run $run: ERROR - CUDA cuBLAS execution failed"
            log_result "CUDA_cuBLAS" "$N" "1" "$run" "0,0,0,0"
        else
            metrics=$(extract_metrics "$output")
            log_result "CUDA_cuBLAS" "$N" "1" "$run" "$metrics"
            time_val=$(echo "$metrics" | awk -F, '{print $4}')
            echo "    Run $run: ${time_val} sec"
        fi
    done
done
echo ""

# ==========================================
# CUDA cuSPARSE
# ==========================================
echo "[6/6] Running CUDA cuSPARSE..."
for N in "${SIZES[@]}"; do
    echo "  N=$N"
    for run in $(seq 1 $NUM_RUNS); do
        output=$(./cg_cusparse $N 2>&1)
        if [ $? -ne 0 ]; then
            echo "    Run $run: ERROR - CUDA cuSPARSE execution failed"
            log_result "CUDA_cuSPARSE" "$N" "1" "$run" "0,0,0,0"
        else
            metrics=$(extract_metrics "$output")
            log_result "CUDA_cuSPARSE" "$N" "1" "$run" "$metrics"
            time_val=$(echo "$metrics" | awk -F, '{print $4}')
            echo "    Run $run: ${time_val} sec"
        fi
    done
done
echo ""
        echo "    Run $run: $(echo $metrics | awk -F, '{print $4}') sec"
echo ""

echo "=========================================="
echo "Experiment Complete!"
echo "Results saved to: $RESULTS_CSV"
echo "=========================================="
echo ""

# ==========================================
# Generate Summary Statistics
# ==========================================
echo "Generating summary statistics..."

SUMMARY_CSV="$OUTPUT_DIR/cg_summary.csv"
echo "Implementation,N,Processors,AvgTime,StdDev,MinTime,MaxTime,AvgIterations,Speedup" > "$SUMMARY_CSV"

# Process results using awk
awk -F, '
NR > 1 {
    key = $1 "," $2 "," $3
    times[key] = times[key] " " $8
    iters[key] = iters[key] " " $5
    count[key]++
}
END {
    for (k in times) {
        n = split(times[k], t)

        # Calculate average time
        sum = 0
        for (i = 1; i <= n; i++) sum += t[i]
        avg = sum / n

        # Calculate std dev
        sumsq = 0
        for (i = 1; i <= n; i++) sumsq += (t[i] - avg)^2
        stddev = sqrt(sumsq / n)

        # Min/Max
        min = t[1]; max = t[1]
        for (i = 2; i <= n; i++) {
            if (t[i] < min) min = t[i]
            if (t[i] > max) max = t[i]
        }

        # Average iterations
        split(iters[k], it)
        isum = 0
        for (i = 1; i <= n; i++) isum += it[i]
        avg_iter = isum / n

        # Store for speedup calculation
        times_avg[k] = avg

        print k "," avg "," stddev "," min "," max "," avg_iter ",0"
    }
}
' "$RESULTS_CSV" | sort -t, -k2,2n -k3,3n > "$SUMMARY_CSV.tmp"

# Calculate speedup (relative to CPU Sequential with same N)
awk -F, '
NR == FNR {
    if ($1 == "CPU_Sequential") baseline[$2] = $6
    next
}
{
    N = $2
    speedup = (baseline[N] > 0) ? baseline[N] / $6 : 1.0
    $9 = sprintf("%.2f", speedup)
    print
}
' OFS=, "$SUMMARY_CSV.tmp" "$SUMMARY_CSV.tmp" > "$SUMMARY_CSV"

rm "$SUMMARY_CSV.tmp"

echo "Summary statistics saved to: $SUMMARY_CSV"
echo ""

# ==========================================
# Display Top Results
# ==========================================
echo "=========================================="
echo "Top 10 Fastest Configurations:"
echo "=========================================="
head -1 "$SUMMARY_CSV"
tail -n +2 "$SUMMARY_CSV" | sort -t, -k6,6n | head -10
echo ""

echo "=========================================="
echo "Best Speedups (vs CPU Sequential):"
echo "=========================================="
head -1 "$SUMMARY_CSV"
tail -n +2 "$SUMMARY_CSV" | sort -t, -k9,9rn | head -10
echo ""

echo "Done! Review results in $OUTPUT_DIR/"
