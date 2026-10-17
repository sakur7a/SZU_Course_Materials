#!/bin/bash
# run_experiments.sh - Run all cache experiments
set -e

echo "========================================="
echo "  Cache Performance Experiments"
echo "========================================="
echo ""

# System info
echo "=== System Info ==="
grep 'model name' /proc/cpuinfo | head -1
grep 'cache size' /proc/cpuinfo | head -1
echo "Cache line: $(getconf LEVEL1_DCACHE_LINESIZE) bytes"
echo ""

# Get CPU frequency
echo "=== Measuring CPU Frequency ==="
python3 -c "
import time, subprocess
# Simple frequency estimation
" 2>/dev/null || true

# Compile all programs
echo "=== Compiling ==="
gcc -O2 -o matrix_bench matrix_bench.c -lm
echo "  matrix_bench: OK"

# Try to compile mountain with original Makefile
make clean 2>/dev/null || true
make
echo "  mountain: OK"

gcc -O2 -o cache_latency cache_latency.c -lm
echo "  cache_latency: OK"

gcc -O2 -o tlb_measure tlb_measure.c -lm
echo "  tlb_measure: OK"

echo ""

# Run matrix benchmark
echo "=== Running Matrix Benchmark ==="
echo "(This may take several minutes for large matrices...)"
./matrix_bench > matrix_results.csv
echo "Saved to matrix_results.csv"
echo ""

# Run mountain (original program)
echo "=== Running Memory Mountain ==="
./mountain > mountain_data_new.txt
echo "Saved to mountain_data_new.txt"
echo ""

# Run cache latency measurement
echo "=== Running Cache Latency Measurement ==="
./cache_latency > /dev/null 2> cache_latency_data.txt
echo "Saved to cache_latency_data.txt"
echo ""

# Run TLB measurement
echo "=== Running TLB Measurement ==="
./tlb_measure > tlb_data.csv 2> tlb_info.txt
echo "Saved to tlb_data.csv"
echo ""

# Generate plots
echo "=== Generating Plots ==="
pip3 install matplotlib numpy 2>/dev/null || pip install matplotlib numpy 2>/dev/null || true
python3 plot_all.py
echo ""

echo "========================================="
echo "  All experiments complete!"
echo "========================================="
