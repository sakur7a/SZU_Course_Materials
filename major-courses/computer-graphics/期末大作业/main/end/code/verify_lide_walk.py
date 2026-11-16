"""验证立德门样板的贴地、连通和避让；需要 numpy、shapely 与 mapbox-earcut。"""
import math

from shapely.geometry import Polygon
from shapely.ops import unary_union

import generate_campus as campus
import verify_scene_data as scene


def projected_mesh(name):
    vertices = scene.mesh_vertices(name)
    triangles = []
    for i in range(0, len(vertices), 3):
        triangle = Polygon([(v[0], v[2]) for v in vertices[i:i+3]])
        if triangle.area > 1e-10:
            triangles.append(triangle)
    if not triangles:
        raise ValueError(f'{name}: no projected triangles')
    return vertices, unary_union(triangles)


walk, footprint = projected_mesh('campus-lide-walk.mesh')
edge, _ = projected_mesh('campus-lide-walk-edge.mesh')
booth = Polygon(campus.LIDE_WALK['gate_booth_footprint_m'])
if footprint.area < 20 or footprint.geom_type != 'Polygon':
    raise ValueError('Lide walk is empty or disconnected')
if footprint.intersection(campus.vehicle_surface).area > 1e-4:
    raise ValueError('Lide walk overlaps a vehicle road')
if footprint.intersection(booth).area > 1e-4:
    raise ValueError('Lide walk overlaps the gate booth')
if any(footprint.intersection(g).area > 1e-4 for _, _, g in campus.buildings):
    raise ValueError('Lide walk overlaps a mapped building')
for name, vertices, height in (('walk', walk, .065), ('edge', edge, .075)):
    errors = [abs(y - scene.terrain_triangle_at(x, z) - height)
              for x, y, z, *_ in vertices]
    if not all(math.isfinite(error) and error < .003 for error in errors):
        raise ValueError(f'Lide {name} does not follow the rendered terrain')
print(f'Lide walk verified: {footprint.area:.1f} m², {len(walk)} walk vertices, {len(edge)} edge vertices.')
