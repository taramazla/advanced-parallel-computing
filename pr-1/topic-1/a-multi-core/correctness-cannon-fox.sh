#!/bin/bash

echo "==============================================================="
echo "          Matrix Multiplication Correctness Test (Results Only)"
echo "==============================================================="

# Check if results directory exists
if [ ! -d "results" ]; then
    echo "ERROR: Results directory not found. Please run the programs first."
    exit 1
fi

# Create a temporary directory for correctness test outputs
mkdir -p correctness_test

# Arrays to store test results
declare -A test_results
declare -A test_status

# Function to extract matrix values from output file
extract_matrix() {
    local file=$1
    # Extract everything after "Result Matrix C:" line and normalize to space-separated values
    awk '
    /Result Matrix C:/{flag=1; next}
    flag {
        # Handle format: c[index] = value
        if (/c\[[0-9]+\] = /) {
            gsub(/c\[[0-9]+\] = /, "")
            print
        }
        # Handle format: space-separated values
        else if (/^[0-9\.\-\+eE ]+$/) {
            # Split line into individual numbers and print each on separate line
            gsub(/[ \t]+/, " ")  # normalize whitespace
            gsub(/^[ \t]+|[ \t]+$/, "")  # trim
            if (length($0) > 0) {
                for (i=1; i<=NF; i++) {
                    print $i
                }
            }
        }
    }' "$file" | grep -E '^[0-9\.\-\+eE]+$' > temp_matrix.txt
}

# Function to compare two matrix files
compare_matrices() {
    local file1=$1
    local file2=$2
    local size=$3

    # Sort both files to handle potential ordering differences
    sort -n "$file1" > temp_sorted1.txt
    sort -n "$file2" > temp_sorted2.txt

    # Compare the sorted matrices
    if diff -q temp_sorted1.txt temp_sorted2.txt > /dev/null; then
        rm -f temp_sorted1.txt temp_sorted2.txt
        return 0  # Matrices are identical
    else
        rm -f temp_sorted1.txt temp_sorted2.txt
        return 1  # Matrices differ
    fi
}

echo "Running correctness tests..."
echo ""

# Test each matrix size
for size in 256 512 1024 2048 4096
do
    echo "Testing matrix size: ${size}x${size}"
    echo "-----------------------------------"

    # Check if reference file exists (1 process)
    reference_file="./results/run-N${size}-P1.out"
    if [ ! -f "$reference_file" ]; then
        echo "ERROR: Reference file not found: $reference_file"
        for procs in 4 16; do
            test_results["${size}_${procs}"]="NO_REF"
            test_status["${size}_${procs}"]="❌"
        done
        continue
    fi

    echo "Using reference: $reference_file"

    # Extract reference matrix
    extract_matrix "$reference_file"
    if [ ! -s temp_matrix.txt ]; then
        echo "ERROR: Could not extract matrix from reference file"
        for procs in 4 16; do
            test_results["${size}_${procs}"]="REF_ERROR"
            test_status["${size}_${procs}"]="❌"
        done
        continue
    fi
    mv temp_matrix.txt "correctness_test/reference_matrix_${size}.txt"

    # Test with different number of processes
    for procs in 4 16
    do
        test_file="./results/run-N${size}-P${procs}.out"
        echo -n "Testing with ${procs} processes... "

        if [ ! -f "$test_file" ]; then
            echo "SKIPPED (file not found)"
            test_results["${size}_${procs}"]="NOT_FOUND"
            test_status["${size}_${procs}"]="⚠️"
            continue
        fi

        # Extract test matrix
        extract_matrix "$test_file"
        if [ ! -s temp_matrix.txt ]; then
            echo "FAILED (could not extract matrix)"
            test_results["${size}_${procs}"]="EXTRACT_ERROR"
            test_status["${size}_${procs}"]="❌"
            continue
        fi
        mv temp_matrix.txt "correctness_test/test_matrix_${size}_${procs}.txt"

        # Compare matrices
        if compare_matrices "correctness_test/reference_matrix_${size}.txt" "correctness_test/test_matrix_${size}_${procs}.txt" ${size}; then
            echo "PASSED ✓"
            test_results["${size}_${procs}"]="PASSED"
            test_status["${size}_${procs}"]="✅"
        else
            echo "FAILED ✗"
            test_results["${size}_${procs}"]="FAILED"
            test_status["${size}_${procs}"]="❌"
        fi
    done

    echo ""
done

# Display results table
echo "==============================================================="
echo "                    CORRECTNESS TEST RESULTS                  "
echo "==============================================================="
echo "Matrix Size | Processors |   Status   |      Result"
echo "------------|------------|------------|------------------"

total_tests=0
passed_tests=0

for size in 256 512 1024 2048 4096
do
    for procs in 4 16
    do
        status_symbol="${test_status[${size}_${procs}]:-⚠️}"
        result_text="${test_results[${size}_${procs}]:-NOT_TESTED}"

        printf "%-11s | %-10s | %-10s | %s\n" "${size}" "${procs}" "${status_symbol}" "${result_text}"

        if [[ "$result_text" != "NOT_TESTED" && "$result_text" != "NOT_FOUND" ]]; then
            total_tests=$((total_tests + 1))
            if [[ "$result_text" == "PASSED" ]]; then
                passed_tests=$((passed_tests + 1))
            fi
        fi
    done
    echo "------------|------------|------------|------------------"
done

# Summary by matrix size
echo ""
echo "==============================================================="
echo "                      SUMMARY BY SIZE                         "
echo "==============================================================="
echo "Matrix Size | Tests Run | Passed | Failed | Success Rate"
echo "------------|-----------|--------|--------|-------------"

for size in 256 512 1024 2048 4096
do
    size_total=0
    size_passed=0

    for procs in 4 16
    do
        result_text="${test_results[${size}_${procs}]:-NOT_TESTED}"
        if [[ "$result_text" != "NOT_TESTED" && "$result_text" != "NOT_FOUND" ]]; then
            size_total=$((size_total + 1))
            if [[ "$result_text" == "PASSED" ]]; then
                size_passed=$((size_passed + 1))
            fi
        fi
    done

    if [ $size_total -gt 0 ]; then
        success_rate=$(( (size_passed * 100) / size_total ))
        printf "%-11s | %-9s | %-6s | %-6s | %s%%\n" "${size}" "${size_total}" "${size_passed}" "$((size_total - size_passed))" "${success_rate}"
    else
        printf "%-11s | %-9s | %-6s | %-6s | %s\n" "${size}" "0" "0" "0" "N/A"
    fi
done

echo ""
echo "==============================================================="
echo "                       OVERALL SUMMARY                        "
echo "==============================================================="

echo "Total tests run: ${total_tests}"
echo "Passed tests: ${passed_tests}"
echo "Failed tests: $((total_tests - passed_tests))"

if [ ${passed_tests} -eq ${total_tests} ] && [ ${total_tests} -gt 0 ]; then
    echo ""
    echo "🎉 ALL TESTS PASSED! The MPI implementation is correct."
elif [ ${total_tests} -eq 0 ]; then
    echo ""
    echo "⚠️  No test files found in results directory."
else
    success_percentage=$(( (passed_tests * 100) / total_tests ))
    echo "Overall success rate: ${success_percentage}%"
    echo ""
    if [ ${success_percentage} -ge 80 ]; then
        echo "✅ Most tests passed, but some issues detected."
    else
        echo "❌ Many tests failed. Please check the implementation."
    fi
fi

echo "==============================================================="

# Legend
echo ""
echo "LEGEND:"
echo "✅ PASSED      - Matrix multiplication results match reference"
echo "❌ FAILED      - Matrix results differ from reference"
echo "❌ NO_REF      - Reference file (1 process) not found"
echo "❌ REF_ERROR   - Could not extract matrix from reference file"
echo "❌ EXTRACT_ERROR - Could not extract matrix from test file"
echo "⚠️  NOT_FOUND   - Test output file not found"

# Cleanup temporary files
rm -f temp_matrix.txt temp_sorted1.txt temp_sorted2.txt

echo ""
echo "Correctness test completed using existing results files."