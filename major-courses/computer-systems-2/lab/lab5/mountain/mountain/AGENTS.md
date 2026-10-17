# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

CSAPP memory hierarchy performance analysis suite. Measures and visualizes cache behavior through memory mountain, matrix multiplication benchmarks, cache latency profiling, and TLB capacity experiments.

## Build Commands

```bash
# Original memory mountain (uses CSAPP timing library)
make

# Individual programs (require Linux, gcc)
gcc -O2 -o matrix_bench matrix_bench.c -lm
gcc -O2 -o cache_latency cache_latency.c -lm
gcc -O2 -o tlb_measure tlb_measure.c -lm

# Standalone matrix benchmarks
gcc -O2 -o matrix_normal matrix_normal.c -lm
gcc -O2 -o matrix_ikj matrix_ikj.c -lm
gcc -O2 -o matrix_block matrix_block.c -lm
```

## Running Experiments

```bash
# Run all experiments and generate plots
./run_experiments.sh

# Individual experiments
./mountain > mountain_data.txt
./matrix_bench > matrix_results.csv
./cache_latency > /dev/null 2> cache_latency_data.txt
./tlb_measure > tlb_data.csv 2> tlb_info.txt

# Plotting (requires matplotlib, numpy)
python3 plot_all.py
```

## Architecture

**Timing Infrastructure:**
- `clock.{c,h}` - CPU frequency measurement via x86 cycle counters
- `fcyc2.{c,h}` - Statistical timing: repeatedly calls test function, returns minimum cycles (from CSAPP)

**Core Programs:**
- `mountain.c` - Memory mountain: sweeps working set size (2KB-64MB) × stride (1-64 elements), outputs throughput in MB/s. Uses `fcyc2` for timing.
- `matrix_bench.c` - Comprehensive matrix multiply benchmark comparing three algorithms across many sizes. Outputs CSV.
- `cache_latency.c` - Uses `rdtsc`/`rdtscp` directly to measure per-access latency and throughput. Outputs memory mountain to stdout, cache hierarchy details to stderr.
- `tlb_measure.c` - Measures TLB capacity by accessing one element per 4KB page. Sequential and random access patterns.

**Matrix Multiply Variants (for cache behavior comparison):**
- `matrix_normal.c` - i-j-k order, poor column-wise access to B
- `matrix_ikj.c` - i-k-j order, B accessed row-wise (better spatial locality)
- `matrix_block.c` - 32×32 blocking (tiling) + i-k-j, fits working set in L1/L2 cache

## Platform Notes

- Linux-only: uses `rdtsc`, `rdtscp`, `clflush`, `gettimeofday`
- WSL2 works; native Windows does not (inline assembly unsupported)
- `cache_latency.c` and `tlb_measure.c` have `#ifdef _WIN32` guards but still require GCC inline asm
- Clock calibration is critical: `mountain.c` calls `mhz(0)` from `clock.c`; the other programs measure it inline

## Data Flow

```
run_experiments.sh
  ├─> matrix_bench      → matrix_results.csv
  ├─> mountain           → mountain_data_new.txt
  ├─> cache_latency      → (stdout) + cache_latency_data.txt (stderr)
  ├─> tlb_measure        → tlb_data.csv + tlb_info.txt (stderr)
  └─> plot_all.py        → PNG plots
```

## Key Constants

- `BLOCK_SIZE = 32` (in matrix_block.c and matrix_bench.c)
- Mountain: `MINBYTES = 2^11`, `MAXBYTES = 2^25`, `MAXSTRIDE = 64`
- TLB: `PAGE_SIZE = 4096`, tests up to 8192 pages (32MB)
