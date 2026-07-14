import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Read the data
data = []
strides = []
sizes = []

with open('mountain_data.txt', 'r') as f:
    lines = f.readlines()
    
    # Read strides from the third line
    strides_line = lines[2].strip().split('\t')
    strides = [int(s[1:]) for s in strides_line if s.startswith('s')]
    
    # Read the rest of the lines
    for line in lines[3:]:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        size_str = parts[0]
        # parse size to bytes
        if size_str.endswith('k'):
            s = int(size_str[:-1]) * 1024
        elif size_str.endswith('m'):
            s = int(size_str[:-1]) * 1024 * 1024
        else:
            continue
        sizes.append(s)
        
        row_data = [float(x) for x in parts[1:]]
        data.append(row_data)

# Reverse to make sizes increasing
sizes.reverse()
data.reverse()

X, Y = np.meshgrid(strides, sizes)
Z = np.array(data)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_xlabel('Stride (words)')
ax.set_ylabel('Size (bytes)')
ax.set_zlabel('Read throughput (MB/s)')
ax.set_title('Memory Mountain')

# Make y-axis logarithmic
ax.set_yscale('log', base=2)

fig.colorbar(surf, shrink=0.5, aspect=0.5)
plt.savefig('mountain.png', dpi=300)
print("Plot saved to mountain.png")
