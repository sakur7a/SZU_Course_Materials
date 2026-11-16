"""Bake the photo-calibrated Lide corridor without rebuilding distant meshes."""
import argparse
import json
import math
import struct
from pathlib import Path

import mapbox_earcut
import numpy as np
from shapely import get_parts
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.prepared import prep

from campus_planting import _mesh_footprint


PLANAR_LAYERS = ('campus-network-sidewalk.mesh', 'campus-network-paths.mesh',
                 'campus-network-verges.mesh')
LAMPS, LAMP_HEADS = 'campus-network-lamps.mesh', 'campus-network-lamp-heads.mesh'
DETAIL_LAYERS = {'campus-network-verge-grass.mesh': 3, 'campus-network-shrubs.mesh': 6,
                 LAMPS: 8, LAMP_HEADS: 16}
STRIDE = 96  # three SZM1 vertices, each with eight floats


def read_triangles(path):
    raw = path.read_bytes()
    count, = struct.unpack_from('<I', raw, 4)
    if raw[:4] != b'SZM1' or count % 3 or len(raw) != 8 + count * 32:
        raise ValueError(f'Invalid mesh: {path}')
    return raw, np.frombuffer(raw, dtype='<f4', offset=8).reshape(-1, 3, 8)


def write_triangles(path, payload):
    if len(payload) % STRIDE or not 0 < len(payload) // 32 <= 500000:
        raise ValueError(f'Invalid output vertex count: {path}')
    path.write_bytes(b'SZM1' + struct.pack('<I', len(payload) // 32) + payload)


def triangulate_on_face(geometry, source, uv_scale=None):
    # Interpolate on the original face, including heights and UVs at cut edges.
    origin = source[0]
    edges = source[1:] - origin
    inverse = np.linalg.inv(edges[:, [0, 2]])
    winding = -1 if np.linalg.det(edges[:, [0, 2]]) < 0 else 1
    for part in get_parts(geometry):
        if part.geom_type != 'Polygon' or part.area < 1e-7:
            continue
        poly = orient(part, sign=winding)
        rings = [np.asarray(ring.coords)[:-1] for ring in (poly.exterior, *poly.interiors)]
        xz = np.concatenate(rings)
        ends = np.cumsum([len(ring) for ring in rings], dtype=np.uint32)
        indices = mapbox_earcut.triangulate_float64(xz, ends)
        vertices = origin + (xz - origin[[0, 2]]) @ inverse @ edges
        vertices[:, [0, 2]] = xz
        if uv_scale is not None:
            vertices[:, 1] += uv_scale[0]
            vertices[:, 6:] = xz / uv_scale[1]
        yield vertices[indices].astype('<f4').tobytes()


def edit_surface(source_path, output_path, mask, *, drape=None):
    raw, triangles = read_triangles(source_path)
    x0, z0, x1, z1 = mask.bounds
    x, z = triangles[:, :, 0], triangles[:, :, 2]
    nearby = ((x.max(1) >= x0) & (x.min(1) <= x1) & (z.max(1) >= z0) & (z.min(1) <= z1))
    prepared = prep(mask)
    output = bytearray()
    changed = 0
    for index, (triangle, near) in enumerate(zip(triangles, nearby)):
        original = memoryview(raw)[8 + index * STRIDE:8 + (index + 1) * STRIDE]
        if near:
            footprint = Polygon(triangle[:, [0, 2]])
            if footprint.area > 1e-9 and prepared.intersects(footprint):
                clipped = footprint.intersection(mask)
                if clipped.area > 1e-7:
                    geometry = clipped if drape else footprint.difference(mask)
                    output.extend(b''.join(triangulate_on_face(geometry, triangle.astype(float), drape)))
                    changed += 1
                    continue
        if not drape:
            output.extend(original)
    write_triangles(output_path, output)
    return changed


def remove_ground_details(source_path, output_path, mask, *, cluster_faces, keep=None):
    raw, triangles = read_triangles(source_path)
    if len(triangles) % cluster_faces:
        raise ValueError(f'Unexpected cluster layout: {source_path}')
    if keep is None:
        centers = triangles.reshape(-1, cluster_faces, 3, 8)[:, :2, :, :].mean((1, 2))
        blocked = prep(mask)
        keep = [not blocked.covers(Point(x, z)) for x, _, z, *_ in centers]
    if len(keep) != len(triangles) // cluster_faces:
        raise ValueError(f'Lamp bodies and heads do not match: {source_path}')
    size = cluster_faces * STRIDE
    output = b''.join(raw[8+i*size:8+(i+1)*size] for i, valid in enumerate(keep) if valid)
    write_triangles(output_path, output)
    return keep


def refresh_roadside_data(source, output, assets):
    """Update the generated inventory from the actual retained lamp and verge meshes."""
    _, posts = read_triangles(assets/LAMPS)
    corners = posts[:, :, [0, 2]].reshape(-1, DETAIL_LAYERS[LAMPS]*3, 2)
    points = ((corners.min(1).astype(float) + corners.max(1))*.5).round(3)
    _, verges = read_triangles(assets/PLANAR_LAYERS[-1])
    edges = (verges[:, 1:, [0, 2]] - verges[:, :1, [0, 2]]).astype(float)
    verge_area = .5 * np.abs(np.linalg.det(edges)).sum()
    inventory = json.loads(source.read_text(encoding='utf-8'))
    inventory['lamp_points_estimate_m'] = points.tolist()
    inventory['verge_area_estimate_m2'] = round(float(verge_area), 2)
    output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return len(points), round(float(verge_area))


def build_layers(*, config, assets, output, road, plaza, buildings, booth):
    """Keep the road mesh intact. Replace only the inferred treatment of the central island."""
    config = json.loads(config.read_text(encoding='utf-8'))
    alignment = config['aerial_alignment']
    anchors = np.asarray(alignment['anchors_xz_uv'])
    controls = np.column_stack((anchors[:, :2], np.ones(len(anchors))))
    transform = np.linalg.lstsq(controls, anchors[:, 2:], rcond=None)[0]
    lawn_observed = (np.asarray(alignment['lawn_corners_uv']) - transform[2]) @ np.linalg.inv(transform[:2])
    residual_px = np.linalg.norm(controls @ transform - anchors[:, 2:], axis=1)
    forbidden = road.buffer(.19).union(plaza)
    island = Polygon(config['island_envelope_xz']).difference(forbidden)
    lawn = Polygon(lawn_observed).intersection(island).difference(road.buffer(config['path_clearance_m']))
    if lawn.geom_type != 'Polygon' or lawn.area < 850:
        raise ValueError('Calibrated lawn is disconnected or implausibly small')
    # A little overlap under the plaza lip closes float rounding gaps between mesh layers.
    walk = island.difference(lawn).union(island.buffer(.018).intersection(plaza))
    if walk.geom_type != 'Polygon' or walk.area < 230 or walk.intersection(road).area > .001:
        raise ValueError('The island sidewalk must be connected and outside the roads')
    # 门岗外侧用一整片低矮绿化，避开已有红砖、人行带和门岗。
    vicinity = box(-432, -44, -386, -21)
    red_walk = _mesh_footprint(assets/'campus-lide-walk.mesh', vicinity)
    gate_paths = unary_union([_mesh_footprint(assets/name, vicinity) for name in
                             ('campus-lide-walk-edge.mesh', *PLANAR_LAYERS[:2])])
    obstacles = unary_union((road, red_walk, gate_paths, booth, *buildings))
    gate_bed = Polygon(config['gate_bed_xz']).difference(obstacles.buffer(.22))
    gate_bed = max(get_parts(gate_bed), key=lambda part: part.area)
    if gate_bed.geom_type != 'Polygon' or gate_bed.area < 12:
        raise ValueError('Gate bed conflicts with the modeled walking space')
    gate_curbs = red_walk.boundary.intersection(box(-422, -35.5, -398, -30.65))
    gate_curbs = gate_curbs.difference(booth.buffer(.34)).difference(road.buffer(.28))
    # 两侧低篱内移，广场和入口的短边保持开敞。
    hedges, lawn_curbs = [], []
    for z0, z1 in ((-60.2, -57.4), (-48.5, -42.6)):
        side = box(-352.0, z0, -281.0, z1)
        for offset, target in ((.72, hedges), (.14, lawn_curbs)):
            line = lawn.buffer(-offset, join_style='mitre').boundary.intersection(side)
            target.extend(list(part.coords) for part in get_parts(line) if part.length > 3)
    if len(hedges) != 2 or len(lawn_curbs) != 2:
        raise ValueError('Expected two continuous hedge and curb runs')
    bed_lanes = unary_union((LineString(((-413, -26.65), (-399, -26.1))),
                             LineString(((-412, -25.8), (-401, -25.15)))))
    bed_lanes = bed_lanes.intersection(gate_bed.buffer(-.42))
    terrain = assets / 'campus-terrain.mesh'
    output.mkdir(parents=True, exist_ok=True)
    edit_surface(terrain, output/'campus-lide-corridor-paving.mesh', walk, drape=(.059, 2.4))
    edit_surface(terrain, output/'campus-lide-corridor-lawn.mesh', lawn, drape=(.051, 3.2))
    edit_surface(terrain, output/'campus-lide-corridor-bed.mesh', gate_bed, drape=(.045, 1.8))
    replaced = {}
    for filename in PLANAR_LAYERS:
        replaced[filename] = edit_surface(assets/filename, output/filename, island.buffer(.008))
    lamp_keep = None
    for filename, faces in DETAIL_LAYERS.items():
        keep = remove_ground_details(assets/filename, output/filename, island.buffer(.6),
                                     cluster_faces=faces, keep=lamp_keep if filename == LAMP_HEADS else None)
        replaced[filename] = len(keep) - sum(keep)
        if filename == LAMPS:
            lamp_keep = keep
    recipe = {
        'reference_fit_max_px': round(float(max(residual_px)), 2),
        'gate_bed': list(gate_bed.exterior.coords),
        'gate_planting': [list(part.coords) for part in get_parts(bed_lanes) if part.length > 1],
        'gate_curbs': [list(part.coords) for part in get_parts(gate_curbs) if part.length > 1],
        'lawn_curbs': lawn_curbs,
        'hedges': hedges,
        'lawn': list(lawn.exterior.coords),
        'island': list(island.exterior.coords),
        'walk_area_m2': round(walk.area, 2),
        'lawn_area_m2': round(lawn.area, 2),
        'gate_bed_area_m2': round(gate_bed.area, 2),
        'locally_edited_records': replaced,
    }
    (output/'campus-lide-corridor-layout.json').write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return recipe


if __name__ == '__main__':
    import generate_campus as campus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=campus.ASSETS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_layers(config=campus.DATA/'lide-corridor.json', assets=args.assets,
                          output=args.output, road=campus.vehicle_surface,
                          plaza=campus.time_square_paving, buildings=[g for _, _, g in campus.buildings],
                          booth=Polygon(campus.LIDE_WALK['gate_booth_footprint_m']))
    refresh_roadside_data(campus.DATA/'roadside-environment.json',
                          args.output/'roadside-environment.json', args.output)
    print({key: value for key, value in result.items() if not isinstance(value, list)})
