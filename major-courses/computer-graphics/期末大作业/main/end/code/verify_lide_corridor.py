"""Check final SZM1 surfaces and compare the local edit against an optional baseline."""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

from campus_lide_corridor import DETAIL_LAYERS, PLANAR_LAYERS, STRIDE, read_triangles
from verify_gate_assets import Terrain


def footprint(path, bounds):
    _, triangles = read_triangles(path)
    x, z = triangles[:, :, 0], triangles[:, :, 2]
    left, bottom, right, top = bounds
    nearby = (x.max(1) >= left) & (x.min(1) <= right) & (z.max(1) >= bottom) & (z.min(1) <= top)
    faces = shapely.polygons(triangles[nearby][:, :, [0, 2]])
    return shapely.union_all(faces[shapely.area(faces) > 1e-8])


def distant_bytes(path, bounds):
    raw, triangles = read_triangles(path)
    x, z = triangles[:, :, 0], triangles[:, :, 2]
    left, bottom, right, top = bounds
    outside = (x.max(1) < left) | (x.min(1) > right) | (z.max(1) < bottom) | (z.min(1) > top)
    data = memoryview(raw)[8:]
    return b''.join(data[i*STRIDE:(i+1)*STRIDE] for i in np.flatnonzero(outside))


def verify(assets, baseline):
    recipe = json.loads((assets/'campus-lide-corridor-layout.json').read_text(encoding='utf-8'))
    island, gate_bed = Polygon(recipe['island']), Polygon(recipe['gate_bed'])
    bounds = island.union(gate_bed).buffer(2).bounds
    height = Terrain(assets)
    layers = {}
    for name, lift in (('paving', .059), ('lawn', .051), ('bed', .045), ('curbs', None), ('leaves', None)):
        path = assets/f'campus-lide-corridor-{name}.mesh'
        _, triangles = read_triangles(path)
        vertices = triangles.reshape(-1, 8)
        if not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError(f'Empty or nonfinite {name}')
        offsets = np.array([y-height(x,z) for x,y,z in vertices[:, :3]])
        if lift is not None and not np.allclose(offsets, lift, atol=.004):
            raise ValueError(f'{name} does not follow rendered terrain')
        if lift is None and (offsets.min() < -.07 or offsets.max() > .74):
            raise ValueError(f'{name} has unsupported height above ground')
        layers[name] = footprint(path, bounds)
    paving, lawn, bed, curb, leaves = (layers[key] for key in ('paving','lawn','bed','curbs','leaves'))
    road = footprint(assets/'campus-road-surface.mesh', bounds)
    red_walk = footprint(assets/'campus-lide-walk.mesh', bounds)
    plaza = footprint(assets/'campus-time-paving.mesh', bounds)
    if paving.buffer(.002).geom_type != 'Polygon' or paving.distance(plaza) > .015:
        raise ValueError('Sidewalk does not connect to the plaza')
    if paving.intersection(road.union(lawn)).area > .025 or bed.intersection(red_walk.union(road)).area > .025:
        raise ValueError('Paving or planting bed overlaps another surface')
    if (not lawn.union(bed).buffer(.025).covers(leaves) or curb.intersection(road).area > .025
            or curb.intersection(paving).area > .025):
        raise ValueError('Curbs or planting enter the protected paths')
    widths = []
    for x in range(-348, -283, 8):
        section = LineString(((x, -70), (x, -35)))
        sections = paving.buffer(.002).intersection(section)
        lengths = sorted(part.length for part in shapely.get_parts(sections) if part.length > .1)
        if len(lengths) != 2 or min(lengths) < 1.80:
            raise ValueError(f'Insufficient two-sided walking space at x={x}')
        widths.append(lengths[0])
    if baseline:
        for name in ('campus-road-surface.mesh','campus-road-curbs.mesh','campus-road-markings.mesh',
                     'campus-terrain.mesh','campus-trees.bin','campus-time-paving.mesh','campus-lide-walk.mesh'):
            if (assets/name).read_bytes() != (baseline/name).read_bytes():
                raise ValueError(f'Protected asset changed: {name}')
        for name in (*PLANAR_LAYERS, *DETAIL_LAYERS):
            extent = island.buffer(1.25).bounds
            if distant_bytes(assets/name, extent) != distant_bytes(baseline/name, extent):
                raise ValueError(f'Nonlocal changes in {name}')
    print(f'Lide corridor verified: connected {paving.area:.1f} m² paving; '
          f'min sampled width {min(widths):.2f} m; {bed.area:.1f} m² gate bed.')
    if baseline:
        print('Protected meshes and nonlocal automatic triangles match the baseline byte for byte.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    verify(args.assets, args.baseline)
