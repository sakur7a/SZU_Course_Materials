"""Compare projected coverage and sampled attributes outside the candidate area.

Only nonvertical surfaces can be represented as height fields; this test does not
prove equality of vertical curb walls or of sub-sample visual details.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Point, box
from shapely.ops import substring
from shapely.strtree import STRtree
import generate_campus as campus


def mesh(path):
    raw = path.read_bytes()
    if raw[:4] != b'SZM1':
        raise ValueError(f'Invalid mesh: {path}')
    n = int.from_bytes(raw[4:8], 'little')
    if n % 3 or len(raw) != 8 + 32 * n:
        raise ValueError(f'Invalid mesh size: {path}')
    return np.frombuffer(raw, dtype='<f4', offset=8).reshape(-1, 3, 8).astype(float)


def projected(triangles):
    xz = triangles[:, :, (0, 2)]
    det = ((xz[:, 1, 0] - xz[:, 0, 0]) * (xz[:, 2, 1] - xz[:, 0, 1])
           - (xz[:, 1, 1] - xz[:, 0, 1]) * (xz[:, 2, 0] - xz[:, 0, 0]))
    mask = abs(det) > 1e-8
    return triangles[mask], shapely.polygons(xz[mask]), int((~mask).sum())


def sample_attributes(source, target, target_polys, roi, limit):
    # Centroids from both meshes, deterministically subsampled, exercise different triangulations.
    centers = source[:, :, (0, 2)].mean(axis=1)
    eligible = np.flatnonzero(~shapely.contains_xy(roi, centers[:, 0], centers[:, 1]))
    if len(eligible) > limit:
        eligible = eligible[np.linspace(0, len(eligible) - 1, limit, dtype=int)]
    tree = STRtree(target_polys)
    max_delta = np.zeros(6)
    compared = missing = ambiguous = 0
    for idx in eligible:
        p = centers[idx]
        hits = tree.query(Point(*p), predicate='intersects')
        if len(hits) != 1:
            missing += not len(hits)
            ambiguous += len(hits) > 1
            continue
        tri = target[int(hits[0])]
        coords = tri[:, (0, 2)]
        matrix = np.array([[coords[0, 0] - coords[2, 0], coords[1, 0] - coords[2, 0]],
                           [coords[0, 1] - coords[2, 1], coords[1, 1] - coords[2, 1]]])
        weights = np.linalg.solve(matrix, p - coords[2])
        weights = np.array([*weights, 1 - weights.sum()])
        a = source[idx][:, (1, 3, 4, 5, 6, 7)].mean(axis=0)
        b = weights @ tri[:, (1, 3, 4, 5, 6, 7)]
        max_delta = np.maximum(max_delta, abs(a - b))
        compared += 1
    return {'compared': compared, 'missing_coverage': int(missing),
            'ambiguous_skipped': int(ambiguous),
            'max_abs_height_m': max_delta[0],
            'max_abs_normal_component': max(max_delta[1:4]),
            'max_abs_uv_component': max(max_delta[4:6])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-assets', type=Path, required=True)
    p.add_argument('--candidate-assets', type=Path, required=True)
    p.add_argument('--padding-m', type=float, default=15)
    p.add_argument('--samples-per-direction', type=int, default=1500)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    entry = json.loads((campus.DATA / 'road-surface-overrides.json').read_text(encoding='utf8'))['overrides'][0]
    line = next(g for key, _, g in campus.ground_vehicle_roads if key == entry['source_way'])
    region = substring(line, *entry['range_m']).buffer(entry['mask_half_width_m'] + args.padding_m)
    roi = box(*region.bounds)  # Same conservative rectangle as audit_road_asset_scope.py.
    result = {'roi_bounds_xz_m': list(roi.bounds), 'layers': {},
              'limitations': 'Projected coverage plus sampled height, normal and UV; vertical faces are excluded, material assignment is per mesh, overlapping projections are skipped.'}
    for name in ('campus-road-surface.mesh', 'campus-road-curbs.mesh', 'campus-network-sidewalk.mesh'):
        old, old_polys, old_vertical = projected(mesh(args.baseline_assets / name))
        new, new_polys, new_vertical = projected(mesh(args.candidate_assets / name))
        old_union = shapely.union_all(old_polys)
        new_union = shapely.union_all(new_polys)
        removed = old_union.difference(new_union).difference(roi)
        added = new_union.difference(old_union).difference(roi)
        row = {'removed_projected_area_m2': removed.area, 'added_projected_area_m2': added.area,
               'changed_coverage_at_cut_m': old_union.symmetric_difference(new_union).intersection(roi.boundary).length,
               'removed_example_xz_m': list(removed.representative_point().coords[0]) if not removed.is_empty else None,
               'added_example_xz_m': list(added.representative_point().coords[0]) if not added.is_empty else None,
               'excluded_vertical_triangles': [old_vertical, new_vertical],
               'old_to_new': sample_attributes(old, new, new_polys, roi, args.samples_per_direction),
               'new_to_old': sample_attributes(new, old, old_polys, roi, args.samples_per_direction)}
        result['layers'][name] = row
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')


if __name__ == '__main__':
    main()
