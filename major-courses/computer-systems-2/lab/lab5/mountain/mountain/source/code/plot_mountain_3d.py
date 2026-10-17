"""
plot_mountain_3d.py - 3D Memory Mountain Visualization
X: stride in bytes (8 bytes per double element)
Y: working set size in bytes (log scale)
Z: read throughput in MB/s
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def parse_mountain_data(data_file):
    sizes_bytes = []
    stride_elems = []
    throughput_data = []

    with open(data_file, 'r') as f:
        lines = f.readlines()

    # Find the stride header line
    stride_line = None
    data_start = 0
    for i, line in enumerate(lines):
        if 's1' in line and 's2' in line:
            stride_line = line.strip()
            data_start = i + 1
            break

    if stride_line is None:
        raise ValueError(f"Could not find stride header in {data_file}")

    # Parse strides (in elements)
    parts = stride_line.split('\t')
    for p in parts:
        p = p.strip()
        if p.startswith('s') and p[1:].isdigit():
            stride_elems.append(int(p[1:]))

    # Convert strides to bytes (each element = sizeof(double) = 8 bytes)
    stride_bytes = [s * 8 for s in stride_elems]

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

        if len(row) >= len(stride_elems):
            sizes_bytes.append(size)
            throughput_data.append(row[:len(stride_elems)])

    # Reverse to make sizes increasing
    sizes_bytes.reverse()
    throughput_data.reverse()

    return sizes_bytes, stride_bytes, throughput_data


def plot_3d_mountain(data_file, output_file='mountain_3d.png'):
    sizes_bytes, stride_bytes, throughput_data = parse_mountain_data(data_file)

    # Use explicit log2 coordinates instead of set_xscale/set_yscale on a 3D
    # axis. Matplotlib's 3D log axes often collapse the perspective and make
    # the surface look like a flat 2D wall.
    X = np.log2(np.array(stride_bytes, dtype=float))
    Y = np.log2(np.array(sizes_bytes, dtype=float))
    Z = np.array(throughput_data)

    # Create meshgrid
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z,
        cmap='turbo',
        edgecolor='black',
        linewidth=0.25,
        alpha=0.96,
        rstride=1,
        cstride=1,
        antialiased=True
    )

    # Labels with requested units
    ax.set_xlabel('Stride (bytes)', fontsize=12, labelpad=10)
    ax.set_ylabel('Working Set Size (bytes)', fontsize=12, labelpad=10)
    ax.set_zlabel('Read Throughput (MB/s)', fontsize=12, labelpad=10)
    ax.set_title('Memory Mountain\n(Read Throughput vs Stride and Working Set Size)',
                 fontsize=14, fontweight='bold')

    # Tick labels show real byte values, while coordinates are log2 values.
    xticks = [8, 16, 32, 64, 128, 256, 512]
    ax.set_xticks(np.log2(xticks))
    ax.set_xticklabels([str(x) for x in xticks])

    yticks = [2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 33554432, 67108864]
    ylabels = ['2KB', '8KB', '32KB', '128KB', '512KB', '2MB', '8MB', '32MB', '64MB']
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(ylabels)

    ax.set_zlim(0, max(26000, float(np.nanmax(Z)) * 1.05))
    ax.set_box_aspect((1.25, 1.05, 0.65))
    ax.grid(True)

    # Colorbar
    fig.colorbar(surf, shrink=0.58, aspect=12, pad=0.08, label='MB/s')

    # Viewing angle chosen to expose the height dimension and cache ridges.
    ax.view_init(elev=28, azim=-135)
    ax.dist = 9

    plt.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.06)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_heatmap(data_file, output_file='mountain_heatmap.png'):
    sizes_bytes, stride_bytes, throughput_data = parse_mountain_data(data_file)

    Z = np.array(throughput_data)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(Z, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    # X-axis: stride in bytes
    ax.set_xticks(range(0, len(stride_bytes), 2))
    ax.set_xticklabels([str(s) for s in stride_bytes[::2]])
    ax.set_xlabel('Stride (bytes)', fontsize=12)

    # Y-axis: size in bytes
    y_labels = []
    for s in sizes_bytes:
        if s >= 1048576:
            y_labels.append(f'{s // 1048576}MB')
        elif s >= 1024:
            y_labels.append(f'{s // 1024}KB')
        else:
            y_labels.append(f'{s}B')

    ax.set_yticks(range(0, len(sizes_bytes), 2))
    ax.set_yticklabels([y_labels[i] for i in range(0, len(sizes_bytes), 2)])
    ax.set_ylabel('Working Set Size (bytes)', fontsize=12)
    ax.set_title('Memory Mountain Heatmap (MB/s)', fontsize=14, fontweight='bold')

    fig.colorbar(im, label='Read Throughput (MB/s)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    import os

    data_file = 'mountain_data_new.txt'
    if not os.path.exists(data_file):
        data_file = 'mountain_data.txt'

    print(f"Using data file: {data_file}")
    print("Generating 3D Memory Mountain...")
    plot_3d_mountain(data_file, 'mountain_3d.png')
    print("Generating Heatmap...")
    plot_heatmap(data_file, 'mountain_heatmap.png')
    print("Done.")
