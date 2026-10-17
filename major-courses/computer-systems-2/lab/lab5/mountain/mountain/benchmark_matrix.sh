#!/bin/bash
# Matrix multiplication benchmark script
# Compiles and runs all three versions with different matrix sizes

SIZES="50 100 200 300 500 800 1000 1500 2000"
RESULTS_FILE="matrix_results.txt"

echo "=== Matrix Multiplication Benchmark ===" > $RESULTS_FILE
echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)" >> $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Compile all versions
gcc -O2 -o matrix_normal matrix_normal.c -lm
gcc -O2 -o matrix_ikj matrix_ikj.c -lm
gcc -O2 -o matrix_block matrix_block.c -lm

printf "%-8s %15s %15s %15s\n" "N" "Normal(ms)" "IKJ(ms)" "Block(ms)" >> $RESULTS_FILE
printf "%-8s %15s %15s %15s\n" "----" "----------" "-------" "---------" >> $RESULTS_FILE

for N in $SIZES; do
    echo "Running N=$N ..."

    # Run each version 3 times, take the best (minimum) time
    normal_min=999999
    for run in 1 2 3; do
        t=$(./matrix_normal $N 2>&1 | grep -oP '[\d.]+(?= ms)')
        if (( $(echo "$t < $normal_min" | bc -l) )); then
            normal_min=$t
        fi
    done

    ikj_min=999999
    for run in 1 2 3; do
        t=$(./matrix_ikj $N 2>&1 | grep -oP '[\d.]+(?= seconds)')
        if (( $(echo "$t < $ikj_min" | bc -l) )); then
            ikj_min=$t
        fi
    done

    block_min=999999
    for run in 1 2 3; do
        t=$(./matrix_block $N 2>&1 | grep -oP '[\d.]+(?= ms)')
        if (( $(echo "$t < $block_min" | bc -l) )); then
            block_min=$t
        fi
    done

    printf "%-8s %15s %15s %15s\n" "$N" "$normal_min" "$ikj_min" "$block_min" >> $RESULTS_FILE
    echo "  N=$N: Normal=${normal_min}ms, IKJ=${ikj_min}s, Block=${block_min}ms"
done

echo "" >> $RESULTS_FILE
echo "=== Benchmark Complete ===" >> $RESULTS_FILE
cat $RESULTS_FILE
