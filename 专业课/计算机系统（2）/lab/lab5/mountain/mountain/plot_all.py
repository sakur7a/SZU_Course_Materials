"""
plot_all.py - Comprehensive visualization for Cache experiments
Generates:
1. Memory Mountain (3D surface plot)
2. Cache latency curves (2D)
3. Matrix multiplication comparison
4. TLB measurement plot
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import sys
import os

# Set font for Chinese characters (if available)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_memory_mountain(data_file, output_file='mountain_3d.png'):
    """Plot 3D memory mountain from mountain data."""
    sizes = []
    strides = []
    throughput_data = []

    with open(data_file, 'r') as f:
        lines = f.readlines()

    # Parse header to get strides
    stride_line = None
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('s') or '\ts1\t' in line:
            stride_line = line.strip()
            data_start = i + 1
            break

    if stride_line is None:
        # Try alternative format
        for i, line in enumerate(lines):
            if 's1' in line and 's2' in line:
                stride_line = line.strip()
                data_start = i + 1
                break

    if stride_line is None:
        print(f"Error: Could not find stride header in {data_file}")
        return

    # Parse strides
    parts = stride_line.split('\t')
    for p in parts:
        p = p.strip()
        if p.startswith('s') and p[1:].isdigit():
            strides.append(int(p[1:]))

    if not strides:
        print("Error: No strides found")
        return

    # Parse data rows
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('CPU'):
            continue
        parts = line.split('\t')
        if len(parts) < 2:
            continue

        size_str = parts[0].strip()
        try:
            if size_str.endswith('m'):
                size = int(size_str[:-1]) * 1024 * 1024
            elif size_str.endswith('k'):
                size = int(size_str[:-1]) * 1024
            else:
                size = int(size_str)
        except ValueError:
            continue

        row = []
        for p in parts[1:]:
            p = p.strip()
            if p:
                try:
                    row.append(float(p))
                except ValueError:
                    row.append(0.0)

        if len(row) >= len(strides):
            sizes.append(size)
            throughput_data.append(row[:len(strides)])

    if not sizes:
        print("Error: No data rows found")
        return

    # Reverse to make sizes increasing
    sizes.reverse()
    throughput_data.reverse()

    X = np.log2(np.array(strides, dtype=float) * 8.0)
    Y = np.log2(np.array(sizes, dtype=float))
    Z = np.array(throughput_data)

    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, Z,
        cmap='turbo',
        edgecolor='black',
        linewidth=0.25,
        alpha=0.96,
        rstride=1,
        cstride=1,
        antialiased=True
    )

    ax.set_xlabel('Stride (bytes)', fontsize=12, labelpad=10)
    ax.set_ylabel('Working Set Size (bytes)', fontsize=12, labelpad=10)
    ax.set_zlabel('Read Throughput (MB/s)', fontsize=12, labelpad=10)
    ax.set_title('Memory Mountain\n(Cache Hierarchy Performance Profile)', fontsize=14, fontweight='bold')

    xticks = [8, 16, 32, 64, 128, 256, 512]
    ax.set_xticks(np.log2(xticks))
    ax.set_xticklabels([str(x) for x in xticks])

    yticks = [2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 33554432]
    ylabels = ['2KB', '8KB', '32KB', '128KB', '512KB', '2MB', '8MB', '32MB']
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(ylabels)

    ax.set_zlim(0, max(26000, float(np.nanmax(Z)) * 1.05))
    ax.set_box_aspect((1.25, 1.05, 0.65))
    ax.view_init(elev=28, azim=-135)

    fig.colorbar(surf, shrink=0.58, aspect=12, pad=0.08, label='MB/s')
    plt.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.06)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    return sizes, strides, throughput_data


def plot_mountain_heatmap(data_file, output_file='mountain_heatmap.png'):
    """Plot memory mountain as a heatmap for clearer visualization."""
    sizes = []
    strides = []
    throughput_data = []

    with open(data_file, 'r') as f:
        lines = f.readlines()

    stride_line = None
    data_start = 0
    for i, line in enumerate(lines):
        if 's1' in line and 's2' in line:
            stride_line = line.strip()
            data_start = i + 1
            break

    parts = stride_line.split('\t')
    for p in parts:
        p = p.strip()
        if p.startswith('s') and p[1:].isdigit():
            strides.append(int(p[1:]))

    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('CPU'):
            continue
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        size_str = parts[0].strip()
        try:
            if size_str.endswith('m'):
                size = int(size_str[:-1])
            elif size_str.endswith('k'):
                size = int(size_str[:-1]) / 1024.0
            else:
                size = int(size_str) / 1024.0
        except ValueError:
            continue

        row = []
        for p in parts[1:]:
            p = p.strip()
            if p:
                try:
                    row.append(float(p))
                except ValueError:
                    row.append(0.0)

        if len(row) >= len(strides):
            sizes.append(size)
            throughput_data.append(row[:len(strides)])

    sizes.reverse()
    throughput_data.reverse()

    Z = np.array(throughput_data)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(Z, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax.set_xticks(range(0, len(strides), 2))
    ax.set_xticklabels([str(s) for s in strides[::2]])
    ax.set_xlabel('Stride (elements)', fontsize=12)

    y_labels = []
    for s in sizes:
        if s >= 1:
            y_labels.append(f'{int(s)}M' if s == int(s) else f'{s:.1f}M')
        else:
            y_labels.append(f'{int(s*1024)}K')

    ax.set_yticks(range(0, len(sizes), 2))
    ax.set_yticklabels([y_labels[i] for i in range(0, len(sizes), 2)])
    ax.set_ylabel('Working Set Size', fontsize=12)
    ax.set_title('Memory Mountain Heatmap (MB/s)', fontsize=14, fontweight='bold')

    fig.colorbar(im, label='Read Throughput (MB/s)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_cache_latency(data_file, output_file='cache_latency.png'):
    """Plot cache latency vs working set size for different strides."""
    sizes_kb = {}
    strides = [1, 2, 4, 8, 16, 32, 64]

    for s in strides:
        sizes_kb[s] = []

    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('Size'):
                continue
            parts = line.split('\t')
            if len(parts) < 8:
                continue
            try:
                size = float(parts[0])
                for i, s in enumerate(strides):
                    lat = float(parts[i+1])
                    sizes_kb[s].append((size, lat))
            except (ValueError, IndexError):
                continue

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf']
    for i, s in enumerate(strides):
        if sizes_kb[s]:
            x = [p[0] for p in sizes_kb[s]]
            y = [p[1] for p in sizes_kb[s]]
            ax.plot(x, y, 'o-', color=colors[i], label=f'Stride={s}', markersize=3, linewidth=1.5)

    # Add cache level annotations
    ax.axvline(x=48, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(x=2048, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(x=36864, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.annotate('L1\n48KB', xy=(48, ax.get_ylim()[1]*0.9), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    ax.annotate('L2\n2MB', xy=(2048, ax.get_ylim()[1]*0.9), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    ax.annotate('L3\n36MB', xy=(36864, ax.get_ylim()[1]*0.9), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Working Set Size (KB)', fontsize=12)
    ax.set_ylabel('Access Latency (cycles)', fontsize=12)
    ax.set_title('Cache Access Latency vs Working Set Size', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_matrix_comparison(data_file, output_file='matrix_comparison.png'):
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
        print("Error: No matrix data found")
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
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_tlb(data_file, output_file='tlb_measurement.png'):
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
        print("Error: No TLB data found")
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
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    print("=== Generating Visualizations ===\n")

    # Memory Mountain
    if os.path.exists('mountain_data.txt'):
        print("1. Memory Mountain 3D...")
        plot_memory_mountain('mountain_data.txt')
        print("2. Memory Mountain Heatmap...")
        plot_mountain_heatmap('mountain_data.txt')
    else:
        print("Warning: mountain_data.txt not found")

    # Cache latency
    if os.path.exists('cache_latency_data.txt'):
        print("3. Cache Latency...")
        plot_cache_latency('cache_latency_data.txt')
    else:
        print("Warning: cache_latency_data.txt not found (run cache_latency first)")

    # Matrix comparison
    if os.path.exists('matrix_results.csv'):
        print("4. Matrix Comparison...")
        plot_matrix_comparison('matrix_results.csv')
    else:
        print("Warning: matrix_results.csv not found (run matrix_bench first)")

    # TLB
    if os.path.exists('tlb_data.csv'):
        print("5. TLB Measurement...")
        plot_tlb('tlb_data.csv')
    else:
        print("Warning: tlb_data.csv not found (run tlb_measure first)")

    print("\n=== Done ===")
