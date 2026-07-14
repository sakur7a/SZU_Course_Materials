"""
plot_experiments.py - Generate all experiment plots:
1. Memory Mountain 3D (stride in bytes)
2. Cache Latency vs Working Set Size
3. Matrix Multiplication Comparison
4. TLB Measurement
"""
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_cache_latency(data_file='cache_latency_data.txt', output='cache_latency.png'):
    """Plot cache latency curves showing cache hierarchy."""
    sizes_kb = []
    data = {s: [] for s in [1, 2, 4, 8, 16, 32, 64]}

    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('Size') or line.startswith('Measured') or line.startswith('Array') or line.startswith('===') or line.startswith('Done'):
                continue
            parts = line.split('\t')
            if len(parts) < 8:
                continue
            try:
                size = float(parts[0])
                sizes_kb.append(size)
                for i, s in enumerate([1, 2, 4, 8, 16, 32, 64]):
                    data[s].append(float(parts[i + 1]))
            except (ValueError, IndexError):
                continue

    if not sizes_kb:
        print("No cache latency data found")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf']

    for i, s in enumerate([1, 2, 4, 8, 16, 32, 64]):
        ax.plot(sizes_kb, data[s], 'o-', color=colors[i], label=f'Stride={s} elem ({s*8}B)',
                markersize=3, linewidth=1.5)

    # Annotate cache levels based on data jumps
    # From data: latency jumps around ~32KB (L1), ~1.5MB (L2), ~32MB (L3)
    ax.axvline(x=32, color='red', linestyle='--', alpha=0.6, linewidth=1)
    ax.axvline(x=1536, color='green', linestyle='--', alpha=0.6, linewidth=1)
    ax.axvline(x=24576, color='blue', linestyle='--', alpha=0.6, linewidth=1)

    ax.annotate('L1 ~32KB', xy=(32, ax.get_ylim()[1] * 0.85), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.annotate('L2 ~1.5MB', xy=(1536, ax.get_ylim()[1] * 0.85), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    ax.annotate('L3 ~24MB', xy=(24576, ax.get_ylim()[1] * 0.85), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Working Set Size (KB)', fontsize=12)
    ax.set_ylabel('Access Latency (cycles)', fontsize=12)
    ax.set_title('Cache Access Latency vs Working Set Size\n(Cache Hierarchy Detection)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_matrix_comparison(data_file='matrix_results.csv', output='matrix_comparison.png'):
    """Plot matrix multiplication performance comparison."""
    sizes = []
    normal_times = []
    ikj_times = []
    block_times = []

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sizes.append(int(row['N']))
                normal_times.append(float(row['Normal_ms']))
                ikj_times.append(float(row['IKJ_ms']))
                block_times.append(float(row['Block_ms']))
            except (KeyError, ValueError):
                continue

    if not sizes:
        print("No matrix data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Absolute time
    ax1.plot(sizes, normal_times, 'rs-', label='Normal (i-j-k)', linewidth=2, markersize=6)
    ax1.plot(sizes, ikj_times, 'go-', label='Optimized (i-k-j)', linewidth=2, markersize=6)
    ax1.plot(sizes, block_times, 'b^-', label='Blocking (32x32)', linewidth=2, markersize=6)
    ax1.set_xlabel('Matrix Size N', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.set_title('Matrix Multiplication: Execution Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    # Plot 2: Speedup relative to normal
    speedup_ikj = [n / o if o > 0 else 0 for n, o in zip(normal_times, ikj_times)]
    speedup_block = [n / b if b > 0 else 0 for n, b in zip(normal_times, block_times)]

    ax2.plot(sizes, speedup_ikj, 'go-', label='IKJ Speedup', linewidth=2, markersize=6)
    ax2.plot(sizes, speedup_block, 'b^-', label='Block Speedup', linewidth=2, markersize=6)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Matrix Size N', fontsize=12)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('Matrix Multiplication: Speedup vs Normal', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_tlb(data_file='tlb_data.csv', output='tlb_measurement.png'):
    """Plot TLB measurement results."""
    pages = []
    sizes_kb = []
    seq_lat = []
    rand_lat = []

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pages.append(int(row['num_pages']))
                sizes_kb.append(float(row['size_KB']))
                seq_lat.append(float(row['sequential_cycles']))
                rand_lat.append(float(row['random_cycles']))
            except (KeyError, ValueError):
                continue

    if not pages:
        print("No TLB data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(sizes_kb, seq_lat, 'bo-', label='Sequential Access', linewidth=2, markersize=4)
    ax1.plot(sizes_kb, rand_lat, 'rs-', label='Random Access', linewidth=2, markersize=4)
    ax1.set_xlabel('Total Pages Accessed (KB)', fontsize=12)
    ax1.set_ylabel('Average Latency (cycles)', fontsize=12)
    ax1.set_title('TLB Latency vs Number of Pages', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)

    # Ratio plot
    ratio = [r / s if s > 0 else 0 for r, s in zip(rand_lat, seq_lat)]
    ax2.plot(sizes_kb, ratio, 'go-', linewidth=2, markersize=4)
    ax2.set_xlabel('Total Pages Accessed (KB)', fontsize=12)
    ax2.set_ylabel('Random/Sequential Latency Ratio', fontsize=12)
    ax2.set_title('TLB Miss Penalty Indicator', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


if __name__ == '__main__':
    print("=== Generating Experiment Plots ===\n")

    if os.path.exists('cache_latency_data.txt'):
        print("1. Cache Latency...")
        plot_cache_latency()
    else:
        print("Warning: cache_latency_data.txt not found")

    if os.path.exists('matrix_results.csv'):
        print("2. Matrix Comparison...")
        plot_matrix_comparison()
    else:
        print("Warning: matrix_results.csv not found")

    if os.path.exists('tlb_data.csv'):
        print("3. TLB Measurement...")
        plot_tlb()
    else:
        print("Warning: tlb_data.csv not found")

    print("\n=== Done ===")
