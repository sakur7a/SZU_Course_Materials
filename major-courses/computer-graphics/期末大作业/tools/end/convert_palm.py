"""将 Quaternius CC0 棕榈 glTF 转为期末场景使用的 SZM1 / RGBA 格式。"""
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image
import trimesh

source, destination = Path(sys.argv[1]), Path(sys.argv[2])
scene = trimesh.load(source, force="scene")
nodes = list(scene.graph.nodes_geometry)
if len(nodes) != 1:
    raise ValueError("Expected one palm mesh")
transform, name = scene.graph[nodes[0]]
mesh = scene.geometry[name]
positions = np.column_stack((mesh.vertices, np.ones(len(mesh.vertices))))
positions = (transform @ positions.T).T[:, :3]
normals = (np.linalg.inv(transform[:3, :3]).T @ mesh.vertex_normals.T).T
normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
indices = mesh.faces.reshape(-1)
vertices = np.column_stack((positions[indices], normals[indices], mesh.visual.uv[indices])).astype("<f4")
with (destination / "palm.mesh").open("wb") as out:
    out.write(b"SZM1")
    out.write(struct.pack("<I", len(vertices)))
    out.write(vertices.tobytes())
image = mesh.visual.material.baseColorTexture.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
with (destination / "palm.rgba").open("wb") as out:
    out.write(struct.pack("<II", image.width, image.height))
    out.write(image.tobytes())
print("palm", len(vertices), "vertices", image.size, "texture")
