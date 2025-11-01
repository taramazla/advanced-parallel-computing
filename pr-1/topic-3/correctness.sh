#!/bin/bash

echo "==============================================================="
echo "    Conjugate Gradient Method Correctness Test (Results Only)"
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

# Function to extract solution vector from output file
extract_solution() {
    local file=$1
    # Extract solution vector x values (only first 10 elements for comparison)
    awk '
    /First 10 elements of solution vector x:/{flag=1; next}
    /\.\.\./{flag=0}
    flag && /x\[/ {
        gsub(/x\[/, "")
        gsub(/\]/, "")
        print $3
    }' "$file" > temp_solution.txt
}

# Function to compare two solution files
compare_solutions() {
    local file1=$1
    local file2=$2
    local tolerance=$3

    # Check if both files exist and have content
    if [ ! -s "$file1" ] || [ ! -s "$file2" ]; then
        return 1
    fi

    # Compare solutions element by element with tolerance
    local max_diff=0
    local line_num=0

    while IFS= read -r val1 && IFS= read -r val2 <&3; do
        line_num=$((line_num + 1))

        # Calculate absolute difference
        diff=$(awk -v v1="$val1" -v v2="$val2" 'BEGIN {
            d = v1 - v2
            if (d < 0) d = -d
            printf "%.15e", d
        }')

        # Track maximum difference
        is_greater=$(awk -v d="$diff" -v m="$max_diff" 'BEGIN {
            if (d > m) print "1"
            else print "0"
        }')

        if [ "$is_greater" -eq 1 ]; then
            max_diff=$diff
        fi
    done < "$file1" 3< "$file2"

    # Check if max difference is within tolerance
    within_tol=$(awk -v d="$max_diff" -v t="$tolerance" 'BEGIN {
        if (d <= t) print "1"
        else print "0"
    }')

    if [ "$within_tol" -eq 1 ]; then
        return 0  # Solutions match within tolerance
    else
        return 1  # Solutions differ
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
        for procs in 2 4 8 16 32; do
            test_results["${size}_${procs}"]="NO_REF"
            test_status["${size}_${procs}"]="❌"
        done
        continue
    fi

    echo "Using reference: $reference_file"

    # Extract reference solution and verification
    extract_solution "$reference_file"
    if [ ! -s temp_solution.txt ]; then
        echo "ERROR: Could not extract solution from reference file"
        for procs in 2 4 8 16 32; do
            test_results["${size}_${procs}"]="REF_ERROR"
            test_status["${size}_${procs}"]="❌"
        done
        continue
    fi
    mv temp_solution.txt "correctness_test/reference_solution_${size}.txt"

    # Extract reference verification status and residual
    ref_verification=$(grep "Verification PASSED\|Verification FAILED" "$reference_file" | awk '{print $2}')
    ref_residual=$(grep "Computed ||b - Ax||" "$reference_file" | awk '{print $4}')
    ref_iterations=$(grep "CG converged in" "$reference_file" | awk '{print $4}')

    echo "Reference (P=1): Iterations=${ref_iterations}, Residual=${ref_residual}, Status=${ref_verification}"

    # Test with different number of processes
    for procs in 2 4 8 16 32
    do
        test_file="./results/run-N${size}-P${procs}.out"
        echo -n "Testing with ${procs} processes... "

        if [ ! -f "$test_file" ]; then
            echo "SKIPPED (file not found)"
            test_results["${size}_${procs}"]="NOT_FOUND"
            test_status["${size}_${procs}"]="⚠️"
            continue
        fi

        # Extract test solution
        extract_solution "$test_file"
        if [ ! -s temp_solution.txt ]; then
            echo "FAILED (could not extract solution)"
            test_results["${size}_${procs}"]="EXTRACT_ERROR"
            test_status["${size}_${procs}"]="❌"
            continue
        fi
        mv temp_solution.txt "correctness_test/test_solution_${size}_${procs}.txt"

        # Extract test verification status
        test_verification=$(grep "Verification PASSED\|Verification FAILED" "$test_file" | awk '{print $2}')
        test_residual=$(grep "Computed ||b - Ax||" "$test_file" | awk '{print $4}')
        test_iterations=$(grep "CG converged in" "$test_file" | awk '{print $4}')

        # Check verification status
        if [ "$test_verification" != "PASSED" ]; then
            echo "FAILED (verification failed: $test_verification)"
            test_results["${size}_${procs}"]="VERIFY_FAILED"
            test_status["${size}_${procs}"]="❌"
            continue
        fi

        # Compare solutions with tolerance
        TOLERANCE=1e-4
        if compare_solutions "correctness_test/reference_solution_${size}.txt" "correctness_test/test_solution_${size}_${procs}.txt" ${TOLERANCE}; then
            echo "PASSED ✓ (Iter=${test_iterations}, Residual=${test_residual})"
            test_results["${size}_${procs}"]="PASSED"
            test_status["${size}_${procs}"]="✅"
        else
            echo "FAILED ✗ (solution differs)"
            test_results["${size}_${procs}"]="FAILED"
            test_status["${size}_${procs}"]="❌"
        fi
    done

    echo ""
done

# Display results table
echo "==============================================================="
echo "                    CORRECTNESS TEST RESULTS                  ".
echo "==============================================================="
echo "Matrix Size | Processors |   Status   |      Result"
echo "------------|------------|------------|------------------"

total_tests=0
passed_tests=0

for size in 256 512 1024 2048 4096
do
    for procs in 2 4 8 16 32
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

    for procs in 2 4 8 16 32
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
    echo "🎉 ALL TESTS PASSED! The CG implementation is correct."
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
echo "✅ PASSED         - CG solution matches reference within tolerance"
echo "❌ FAILED         - CG solution differs from reference"
echo "❌ VERIFY_FAILED  - Verification check (||b - Ax||) failed"
echo "❌ NO_REF         - Reference file (1 process) not found"
echo "❌ REF_ERROR      - Could not extract solution from reference file"
echo "❌ EXTRACT_ERROR  - Could not extract solution from test file"
echo "⚠️  NOT_FOUND     - Test output file not found"

# Cleanup temporary files
rm -f temp_solution.txt

echo ""
echo "Correctness test completed using existing results files."
