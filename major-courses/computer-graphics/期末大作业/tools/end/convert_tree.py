"""将 Quaternius CC0 树木 GLB 转为本作业的简洁三角网格及 RGBA 纹理。

使用方式：python convert_tree.py <Tree.glb> <assets-directory>
离线工具需要 trimesh、numpy 和 Pillow；最终可执行程序不需要 Python。
"""
from pathlib import Path
import struct
import sys

import numpy as np
import trimesh


source = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
scene = trimesh.load(source, force="scene")

for node in scene.graph.nodes_geometry:
    transform, geometry_name = scene.graph[node]
    geometry = scene.geometry[geometry_name]
    material = geometry.visual.material
    label = "tree-leaves" if "Leaves" in material.name else "tree-bark"
    positions = np.column_stack((geometry.vertices, np.ones(len(geometry.vertices))))
    positions = (transform @ positions.T).T[:, :3]
    normal_matrix = np.linalg.inv(transform[:3, :3]).T
    normals = (normal_matrix @ geometry.vertex_normals.T).T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    uvs = geometry.visual.uv
    indices = geometry.faces.reshape(-1)
    vertices = np.column_stack((positions[indices], normals[indices], uvs[indices])).astype("<f4")
    with (destination / (label + ".mesh")).open("wb") as output:
        output.write(b"SZM1")
        output.write(struct.pack("<I", len(vertices)))
        output.write(vertices.tobytes())
    # trimesh exposes V coordinates in bottom-left convention; OpenGL data
    # starts with the lower row, so flip the top-down image bytes for upload.
    from PIL import Image
    image = material.baseColorTexture.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    with (destination / (label + ".rgba")).open("wb") as output:
        output.write(struct.pack("<II", image.width, image.height))
        output.write(image.tobytes())
    print(label, "vertices", len(vertices), "texture", image.size)
