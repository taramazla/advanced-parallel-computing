#!/bin/bash

# Script untuk membuat ringkasan tabel dari hasil eksperimen CG
# Input: cg_results.csv
# Output: Berbagai tabel summary dalam format TXT

INPUT_CSV="experiment_results/cg_results.csv"
OUTPUT_DIR="summary_tables"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  Creating Summary Tables"
echo "=========================================="
echo ""

# ==========================================
# 1. AVERAGE TIME TABLE (by Implementation and N)
# ==========================================
echo "[1] Creating Average Time Summary..."

cat > "$OUTPUT_DIR/1_avg_time_summary.txt" << 'EOF'
================================================================================
                    AVERAGE EXECUTION TIME SUMMARY
================================================================================

Table 1: Average Time (seconds) - Lower is Better
--------------------------------------------------------------------------------
EOF

awk -F, '
NR > 1 {
    key = $1 "," $2 "," $3
    times[key] += $8
    count[key]++
    impl[$1] = 1
    sizes[$2] = 1
    procs[$1 "," $3] = 1
}
END {
    # Print header
    printf "%-20s | %-6s | %-10s | %-12s | %-8s\n", "Implementation", "Size", "Processors", "Avg Time (s)", "Speedup"
    printf "%-20s-+-%-6s-+-%-10s-+-%-12s-+-%-8s\n", "--------------------", "------", "----------", "------------", "--------"

    # Calculate baseline (CPU Sequential)
    for (key in times) {
        split(key, k, ",")
        if (k[1] == "CPU_Sequential") {
            baseline[k[2]] = times[key] / count[key]
        }
    }

    # Print all results sorted
    for (key in times) {
        split(key, k, ",")
        avg = times[key] / count[key]
        speedup = (baseline[k[2]] > 0) ? baseline[k[2]] / avg : 1.0
        printf "%-20s | %-6s | %-10s | %12.6f | %8.2fx\n", k[1], k[2], k[3], avg, speedup
    }
}
' "$INPUT_CSV" | sort -t'|' -k2,2n -k1,1 >> "$OUTPUT_DIR/1_avg_time_summary.txt"

echo "✓ Average Time Summary: $OUTPUT_DIR/1_avg_time_summary.txt"

# ==========================================
# 2. SPEEDUP TABLE (Relative to CPU Sequential)
# ==========================================
echo "[2] Creating Speedup Table..."

cat > "$OUTPUT_DIR/2_speedup_table.txt" << 'EOF'
================================================================================
                        SPEEDUP vs CPU SEQUENTIAL
================================================================================

Table 2: Speedup Comparison (Higher is Better)
--------------------------------------------------------------------------------
EOF

awk -F, '
NR > 1 {
    key = $1 "," $2 "," $3
    times[key] += $8
    count[key]++
}
END {
    # Calculate baseline
    for (key in times) {
        split(key, k, ",")
        if (k[1] == "CPU_Sequential") {
            baseline[k[2]] = times[key] / count[key]
        }
    }

    # Header
    printf "\n%-20s | N=1024 | N=2048 | N=4096\n", "Implementation"
    printf "%-20s-+--------+--------+--------\n", "--------------------"

    # Collect data by implementation
    for (key in times) {
        split(key, k, ",")
        impl = k[1]
        n = k[2]
        proc = k[3]
        avg = times[key] / count[key]
        speedup = (baseline[n] > 0) ? baseline[n] / avg : 1.0

        data[impl, n, proc] = speedup
        impls[impl] = 1
        configs[impl, proc] = 1
    }

    # Print by implementation
    for (impl in impls) {
        for (proc in configs) {
            split(proc, p, SUBSEP)
            if (p[1] == impl) {
                printf "%-20s | %6.2fx | %6.2fx | %6.2fx   [%s]\n",
                    impl,
                    data[impl, "1024", p[2]] ? data[impl, "1024", p[2]] : 0,
                    data[impl, "2048", p[2]] ? data[impl, "2048", p[2]] : 0,
                    data[impl, "4096", p[2]] ? data[impl, "4096", p[2]] : 0,
                    p[2]
            }
        }
    }
}
' "$INPUT_CSV" >> "$OUTPUT_DIR/2_speedup_table.txt"

echo "✓ Speedup Table: $OUTPUT_DIR/2_speedup_table.txt"

# ==========================================
# 3. BEST PERFORMANCE TABLE
# ==========================================
echo "[3] Creating Best Performance Table..."

cat > "$OUTPUT_DIR/3_best_performance.txt" << 'EOF'
================================================================================
                        BEST PERFORMANCE RANKING
================================================================================

Table 3: Top 15 Fastest Configurations
--------------------------------------------------------------------------------
EOF

awk -F, '
NR > 1 {
    key = $1 "," $2 "," $3
    times[key] += $8
    count[key]++
}
END {
    printf "%-5s | %-20s | %-6s | %-10s | %-12s\n", "Rank", "Implementation", "Size", "Processors", "Avg Time (s)"
    printf "%-5s-+-%-20s-+-%-6s-+-%-10s-+-%-12s\n", "-----", "--------------------", "------", "----------", "------------"

    # Calculate averages and sort
    n_entries = 0
    for (key in times) {
        avg[key] = times[key] / count[key]
        keys[n_entries++] = key
    }

    # Bubble sort by average time
    for (i = 0; i < n_entries; i++) {
        for (j = i + 1; j < n_entries; j++) {
            if (avg[keys[i]] > avg[keys[j]]) {
                tmp = keys[i]
                keys[i] = keys[j]
                keys[j] = tmp
            }
        }
    }

    # Print top 15
    for (i = 0; i < (n_entries < 15 ? n_entries : 15); i++) {
        split(keys[i], k, ",")
        printf "%5d | %-20s | %6s | %10s | %12.6f\n", i+1, k[1], k[2], k[3], avg[keys[i]]
    }
}
' "$INPUT_CSV" >> "$OUTPUT_DIR/3_best_performance.txt"

echo "✓ Best Performance Table: $OUTPUT_DIR/3_best_performance.txt"

# ==========================================
# 4. SCALABILITY ANALYSIS
# ==========================================
echo "[4] Creating Scalability Analysis..."

cat > "$OUTPUT_DIR/4_scalability_analysis.txt" << 'EOF'
================================================================================
                         SCALABILITY ANALYSIS
================================================================================

Table 4a: OpenMP Scalability (by Thread Count)
--------------------------------------------------------------------------------
EOF

awk -F, '
$1 == "OpenMP_Multicore" {
    key = $2 "," $3
    times[key] += $8
    count[key]++
    sizes[$2] = 1
    procs[$3] = 1
}
END {
    printf "%-6s | %-10s | %-12s | %-10s | %-10s\n", "Size", "Threads", "Avg Time (s)", "Speedup", "Efficiency"
    printf "%-6s-+-%-10s-+-%-12s-+-%-10s-+-%-10s\n", "------", "----------", "------------", "----------", "----------"

    for (size in sizes) {
        base_time = 0
        for (proc in procs) {
            key = size "," proc
            if (key in times) {
                avg = times[key] / count[key]
                if (proc == "2") base_time = avg
                speedup = (base_time > 0) ? base_time / avg : 1.0
                threads = proc + 0
                efficiency = speedup / threads * 100
                printf "%6s | %10s | %12.6f | %10.2fx | %9.1f%%\n", size, proc, avg, speedup, efficiency
            }
        }
        printf "%-6s-+-%-10s-+-%-12s-+-%-10s-+-%-10s\n", "------", "----------", "------------", "----------", "----------"
    }
}
' "$INPUT_CSV" >> "$OUTPUT_DIR/4_scalability_analysis.txt"

cat >> "$OUTPUT_DIR/4_scalability_analysis.txt" << 'EOF'

Table 4b: MPI Scalability (by Process Count)
--------------------------------------------------------------------------------
EOF

awk -F, '
$1 == "MPI_Distributed" {
    key = $2 "," $3
    times[key] += $8
    count[key]++
    sizes[$2] = 1
    procs[$3] = 1
}
END {
    printf "%-6s | %-10s | %-12s | %-10s | %-10s\n", "Size", "Processes", "Avg Time (s)", "Speedup", "Efficiency"
    printf "%-6s-+-%-10s-+-%-12s-+-%-10s-+-%-10s\n", "------", "----------", "------------", "----------", "----------"

    for (size in sizes) {
        base_time = 0
        for (proc in procs) {
            key = size "," proc
            if (key in times) {
                avg = times[key] / count[key]
                if (proc == "2") base_time = avg
                speedup = (base_time > 0) ? base_time / avg : 1.0
                processes = proc + 0
                efficiency = speedup / processes * 100
                printf "%6s | %10s | %12.6f | %10.2fx | %9.1f%%\n", size, proc, avg, speedup, efficiency
            }
        }
        printf "%-6s-+-%-10s-+-%-12s-+-%-10s-+-%-10s\n", "------", "----------", "------------", "----------", "----------"
    }
}
' "$INPUT_CSV" >> "$OUTPUT_DIR/4_scalability_analysis.txt"

echo "✓ Scalability Analysis: $OUTPUT_DIR/4_scalability_analysis.txt"

# ==========================================
# 5. PLATFORM COMPARISON
# ==========================================
echo "[5] Creating Platform Comparison..."

cat > "$OUTPUT_DIR/5_platform_comparison.txt" << 'EOF'
================================================================================
                      PLATFORM COMPARISON SUMMARY
================================================================================

Table 5: Best Time per Platform (seconds)
--------------------------------------------------------------------------------
EOF

awk -F, '
NR > 1 {
    platform = ""
    if ($1 == "CPU_Sequential") platform = "CPU"
    else if ($1 == "OpenMP_Multicore") platform = "OpenMP"
    else if ($1 == "MPI_Distributed") platform = "MPI"
    else if ($1 ~ /CUDA/) platform = $1

    if (platform != "") {
        key = platform "," $2
        if (!(key in best_time) || $8 < best_time[key]) {
            best_time[key] = $8
            best_config[key] = $3
        }
        platforms[platform] = 1
        sizes[$2] = 1
    }
}
END {
    printf "%-20s | %-6s | %-12s | %-10s\n", "Platform", "Size", "Best Time", "Config"
    printf "%-20s-+-%-6s-+-%-12s-+-%-10s\n", "--------------------", "------", "------------", "----------"

    for (platform in platforms) {
        for (size in sizes) {
            key = platform "," size
            if (key in best_time) {
                printf "%-20s | %6s | %12.6f | %10s\n", platform, size, best_time[key], best_config[key]
            }
        }
    }

    printf "\n\nSpeedup Summary (vs CPU Sequential):\n"
    printf "%-20s-+-%-6s-+-%-12s\n", "--------------------", "------", "------------"

    # Get CPU baseline
    for (size in sizes) {
        cpu_key = "CPU," size
        if (cpu_key in best_time) {
            cpu_baseline[size] = best_time[cpu_key]
        }
    }

    for (platform in platforms) {
        if (platform != "CPU") {
            for (size in sizes) {
                key = platform "," size
                if (key in best_time && size in cpu_baseline) {
                    speedup = cpu_baseline[size] / best_time[key]
                    printf "%-20s | %6s | %11.2fx\n", platform, size, speedup
                }
            }
        }
    }
}
' "$INPUT_CSV" >> "$OUTPUT_DIR/5_platform_comparison.txt"

echo "✓ Platform Comparison: $OUTPUT_DIR/5_platform_comparison.txt"

# ==========================================
# 6. DETAILED STATISTICS
# ==========================================
echo "[6] Creating Detailed Statistics..."

cat > "$OUTPUT_DIR/6_detailed_statistics.txt" << 'EOF'
================================================================================
                        DETAILED STATISTICS
================================================================================

Table 6: Statistical Summary (3 runs per configuration)
--------------------------------------------------------------------------------
EOF

awk -F, '
NR > 1 {
    key = $1 "," $2 "," $3
    times[key, ++n[key]] = $8
}
END {
    printf "%-20s | %-6s | %-6s | %-8s | %-8s | %-8s | %-8s\n",
        "Implementation", "Size", "Procs", "Min", "Max", "Avg", "StdDev"
    printf "%-20s-+-%-6s-+-%-6s-+-%-8s-+-%-8s-+-%-8s-+-%-8s\n",
        "--------------------", "------", "------", "--------", "--------", "--------", "--------"

    for (key in n) {
        count = n[key]

        # Calculate min, max, avg
        min = times[key, 1]
        max = times[key, 1]
        sum = 0
        for (i = 1; i <= count; i++) {
            val = times[key, i]
            sum += val
            if (val < min) min = val
            if (val > max) max = val
        }
        avg = sum / count

        # Calculate stddev
        sumsq = 0
        for (i = 1; i <= count; i++) {
            diff = times[key, i] - avg
            sumsq += diff * diff
        }
        stddev = sqrt(sumsq / count)

        split(key, k, ",")
        printf "%-20s | %6s | %6s | %8.6f | %8.6f | %8.6f | %8.6f\n",
            k[1], k[2], k[3], min, max, avg, stddev
    }
}
' "$INPUT_CSV" >> "$OUTPUT_DIR/6_detailed_statistics.txt"

echo "✓ Detailed Statistics: $OUTPUT_DIR/6_detailed_statistics.txt"

# ==========================================
# 7. CREATE MASTER SUMMARY
# ==========================================
echo "[7] Creating Master Summary..."

cat > "$OUTPUT_DIR/0_MASTER_SUMMARY.txt" << EOF
================================================================================
           CONJUGATE GRADIENT PERFORMANCE EXPERIMENT SUMMARY
================================================================================

Experiment Date: $(date +"%Y-%m-%d %H:%M:%S")
Input File: $INPUT_CSV
Total Configurations Tested: $(tail -n +2 "$INPUT_CSV" | wc -l)

--------------------------------------------------------------------------------
                            KEY FINDINGS
--------------------------------------------------------------------------------

EOF

# Find overall fastest
FASTEST=$(awk -F, 'NR>1 {key=$1","$2","$3; times[key]+=$8; count[key]++}
END {
    min_time = 999999; min_key = ""
    for (key in times) {
        avg = times[key]/count[key]
        if (avg < min_time) {min_time = avg; min_key = key}
    }
    split(min_key, k, ",")
    printf "%-20s | N=%-6s | Procs=%-6s | Time=%8.6f sec\n", k[1], k[2], k[3], min_time
}' "$INPUT_CSV")

echo "FASTEST CONFIGURATION:" >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt"
echo "$FASTEST" >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt"
echo "" >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt"

# Best speedup
BEST_SPEEDUP=$(awk -F, '
NR>1 {
    key=$1","$2","$3; times[key]+=$8; count[key]++
}
END {
    for (key in times) {
        split(key, k, ",")
        if (k[1] == "CPU_Sequential") baseline[k[2]] = times[key]/count[key]
    }
    max_speedup = 0; max_key = ""
    for (key in times) {
        split(key, k, ",")
        avg = times[key]/count[key]
        if (baseline[k[2]] > 0) {
            speedup = baseline[k[2]]/avg
            if (speedup > max_speedup) {max_speedup = speedup; max_key = key}
        }
    }
    split(max_key, k, ",")
    printf "%-20s | N=%-6s | Procs=%-6s | Speedup=%6.2fx\n", k[1], k[2], k[3], max_speedup
}' "$INPUT_CSV")

echo "BEST SPEEDUP vs CPU:" >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt"
echo "$BEST_SPEEDUP" >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt"
echo "" >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt"

cat >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt" << 'EOF'
--------------------------------------------------------------------------------
                        AVAILABLE SUMMARY FILES
--------------------------------------------------------------------------------

1. 0_MASTER_SUMMARY.txt          - This file (overview)
2. 1_avg_time_summary.txt        - Average execution times
3. 2_speedup_table.txt           - Speedup comparison table
4. 3_best_performance.txt        - Top 15 fastest configurations
5. 4_scalability_analysis.txt   - OpenMP and MPI scalability
6. 5_platform_comparison.txt    - Platform-by-platform comparison
7. 6_detailed_statistics.txt    - Min/Max/Avg/StdDev statistics

--------------------------------------------------------------------------------
                            QUICK STATISTICS
--------------------------------------------------------------------------------

EOF

# Count configurations per platform
awk -F, 'NR>1 {configs[$1]++} END {
    printf "Configurations tested per platform:\n"
    for (impl in configs) printf "  - %-20s: %3d runs\n", impl, configs[impl]
}' "$INPUT_CSV" >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt"

echo "" >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt"
echo "Matrix sizes tested: 1024, 2048, 4096" >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt"
echo "Runs per configuration: 3 (averaged)" >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt"

cat >> "$OUTPUT_DIR/0_MASTER_SUMMARY.txt" << 'EOF'

================================================================================
                              END OF SUMMARY
================================================================================
EOF

echo "✓ Master Summary: $OUTPUT_DIR/0_MASTER_SUMMARY.txt"

# ==========================================
# FINAL OUTPUT
# ==========================================
echo ""
echo "=========================================="
echo "  Summary Tables Created Successfully!"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.txt | awk '{printf "  - %s (%s)\n", $9, $5}'
echo ""
echo "To view master summary:"
echo "  cat $OUTPUT_DIR/0_MASTER_SUMMARY.txt"
echo ""
echo "To view all summaries:"
echo "  less $OUTPUT_DIR/*.txt"
echo ""
echo "=========================================="
