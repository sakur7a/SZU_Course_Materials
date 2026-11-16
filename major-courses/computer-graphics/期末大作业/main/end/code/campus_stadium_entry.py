"""Patch the photographed northwest stadium opening without rebuilding campus.

All dimensions and the alignment are model estimates; see the adjacent data
and source notes.  Read-only inputs are the original stands, rendered terrain,
road and field meshes.  No full-campus generator is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import mapbox_earcut
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "data" / "stadium-entry.json"
DEFAULT_BASE = HERE / "data" / "stadium-entry-baseline"
DEFAULT_OUTPUT = HERE.parents[2] / "build" / "stadium-entry-candidate" / "assets"
DEFAULT_ASSETS = HERE / "assets"
_terrain_path = DEFAULT_ASSETS / "campus-terrain.mesh"
_road_path = DEFAULT_ASSETS / "campus-road-surface.mesh"
_field_path = DEFAULT_ASSETS / "campus-stadium-surface.mesh"
STAND_NAMES = ("campus-stands-base.mesh", "campus-stands-blue.mesh", "campus-stands-lines.mesh")
PROTECTED_NAMES = ("campus-stands-red.mesh", "campus-stands-orange.mesh", "campus-stands-yellow.mesh",
                   "campus-stadium-canopy.mesh", "campus-stadium-masts.mesh", "campus-stadium-lamps.mesh",
                   "campus-stadium-goals.mesh", "stadium-surface.ppm")


def _read_mesh(path):
    data = Path(path).read_bytes()
    if len(data) < 8 or data[:4] != b"SZM1":
        raise ValueError(f"Invalid mesh: {path}")
    count, = struct.unpack_from("<I", data, 4)
    if count < 3 or count % 3 or len(data) != 8 + count * 32:
        raise ValueError(f"Invalid mesh length: {path}")
    return np.frombuffer(data, dtype="<f4", offset=8).reshape(-1, 3, 8).copy()


def _save_mesh(path, triangles):
    values = np.asarray(triangles, dtype="<f4").reshape(-1, 3, 8)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Empty or invalid mesh: {path}")
    Path(path).write_bytes(b"SZM1" + struct.pack("<I", len(values) * 3) + values.tobytes())
    return len(values) * 3


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Frame:
    def __init__(self, cfg):
        self.origin = np.asarray(cfg["origin"], dtype=float)
        direction = np.asarray(cfg["inside_direction"], dtype=float)
        if self.origin.shape != (2,) or direction.shape != (2,) or np.linalg.norm(direction) < 1e-8:
            raise ValueError("Invalid stadium entry frame")
        self.tangent = direction / np.linalg.norm(direction)
        self.across = np.array((-self.tangent[1], self.tangent[0]))

    def xz(self, along, across):
        return self.origin + along * self.tangent + across * self.across

    def local(self, xz):
        delta = np.asarray(xz, dtype=float) - self.origin
        return np.stack((delta @ self.tangent, delta @ self.across), axis=-1)

    def rectangle(self, start, end, half_width):
        return Polygon([self.xz(start, -half_width), self.xz(end, -half_width),
                        self.xz(end, half_width), self.xz(start, half_width)])


def _pieces(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from _pieces(child)


def _line_parts(geometry):
    if geometry.geom_type == "LineString":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from _line_parts(child)


def _cross2(a, b):
    return a[0]*b[1] - a[1]*b[0]


def _triangulate(poly):
    rings = [list(poly.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in poly.interiors]
    rings = [ring for ring in rings if len(ring) >= 3]
    if not rings:
        return
    coords = np.asarray([p for ring in rings for p in ring], dtype=np.float64)
    ends = np.cumsum([len(ring) for ring in rings], dtype=np.uint32)
    ids = mapbox_earcut.triangulate_float64(coords, ends)
    for tri in coords[ids].reshape(-1, 3, 2):
        if abs(_cross2(tri[1] - tri[0], tri[2] - tri[0])) > 1e-10:
            yield tri


def _weights(xz, x, z):
    a, b, c = xz
    den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
    if abs(den) < 1e-12:
        raise ValueError("Degenerate projected triangle")
    u = ((b[1]-c[1])*(x-c[0]) + (c[0]-b[0])*(z-c[1])) / den
    v = ((c[1]-a[1])*(x-c[0]) + (a[0]-c[0])*(z-c[1])) / den
    return np.array((u, v, 1-u-v))


class Surface:
    """The actual rendered triangles near the entry, with exact planar heights."""
    def __init__(self, path, selection):
        all_faces = _read_mesh(path)
        x0, z0, x1, z1 = selection.bounds
        a = all_faces
        mask = ((a[:,:,0].max(axis=1) >= x0) & (a[:,:,0].min(axis=1) <= x1)
                & (a[:,:,2].max(axis=1) >= z0) & (a[:,:,2].min(axis=1) <= z1)
                & (a[:,:,4].min(axis=1) >= .7))
        faces, polygons = [], []
        for tri in a[mask]:
            poly = Polygon(tri[:, (0, 2)])
            if poly.area > 1e-8 and poly.intersects(selection):
                faces.append(tri)
                polygons.append(poly)
        if not faces:
            raise ValueError(f"No visible surface near entry in {path}")
        self.faces = faces
        self.polygons = polygons
        self.domain = unary_union(polygons)

    def height(self, x, z, required=True):
        values = []
        for tri, poly in zip(self.faces, self.polygons):
            x0, z0, x1, z1 = poly.bounds
            if not (x0-1e-5 <= x <= x1+1e-5 and z0-1e-5 <= z <= z1+1e-5):
                continue
            w = _weights(tri[:, (0, 2)], x, z)
            if min(w) >= -1e-5:
                values.append(float(w @ tri[:, 1]))
        if not values:
            if required:
                raise ValueError(f"Surface missing at {(x, z)}")
            return None
        if max(values) - min(values) > .04:
            raise ValueError(f"Overlapping surface heights at {(x, z)}: {values}")
        return max(values)


def _clip_half(poly, values, positive):
    """Clip 3D vertices, interpolating position, normal and UV together."""
    result = []
    for i, current in enumerate(poly):
        previous = poly[i-1]
        ca, pa = values[i], values[i-1]
        current_in = ca >= 0 if positive else ca <= 0
        previous_in = pa >= 0 if positive else pa <= 0
        if current_in != previous_in:
            t = pa / (pa-ca)
            result.append(previous + t * (current-previous))
        if current_in:
            result.append(current)
    return result


def _nondegenerate(a, b, c):
    return np.linalg.norm(np.cross(b[:3]-a[:3], c[:3]-a[:3])) > 1e-8


def _cut_stands(triangles, frame, opening):
    """Difference against the convex slot, including vertical rail triangles."""
    lo, hi = opening["along_m"]
    half = opening["half_width_m"]
    kept, changed = [], 0
    for original in triangles:
        # Triangles wholly outside stay byte-for-byte identical when saved.
        remaining = [list(np.asarray(original, dtype=float))]
        outside = []
        for axis, limit, sign in ((0, lo, 1), (0, hi, -1), (1, -half, 1), (1, half, -1)):
            next_remaining = []
            for poly in remaining:
                local = frame.local(np.asarray(poly)[:, (0, 2)])
                values = sign * (local[:, axis]-limit)
                if np.min(values) >= -1e-9:
                    next_remaining.append(poly)
                elif np.max(values) <= 1e-9:
                    outside.append(poly)
                else:
                    part_out = _clip_half(poly, values, False)
                    part_in = _clip_half(poly, values, True)
                    if len(part_out) >= 3:
                        outside.append(part_out)
                    if len(part_in) >= 3:
                        next_remaining.append(part_in)
            remaining = next_remaining
            if not remaining:
                break
        if not remaining:
            # No part was removed; preserve the original data and ordering.
            kept.append(original)
            continue
        changed += 1
        for poly in outside:
            for i in range(1, len(poly)-1):
                if _nondegenerate(poly[0], poly[i], poly[i+1]):
                    kept.append(np.asarray((poly[0], poly[i], poly[i+1])))
    if not changed:
        raise ValueError("Opening did not intersect an expected stand layer")
    return kept, changed


def _remove_foldback(original, spec):
    # 删除前核对基线中实际要处理的面和被保留的相邻面，避免仅凭序号误删。
    checks = ((STAND_NAMES[0], 0, 2, "base_piece_sha256"),
              (STAND_NAMES[1], 0, 2, "blue_piece_sha256"),
              (STAND_NAMES[2], 0, 16, "lines_piece_sha256"),
              (STAND_NAMES[0], 92, 2, "retained_piece_sha256"))
    for name, start, count, key in checks:
        part = original[name][start:start+count]
        if len(part) != count or hashlib.sha256(part.tobytes()).hexdigest() != spec[key]:
            raise ValueError(f"Stadium baseline geometry changed: {name} triangle {start}")
    return {STAND_NAMES[0]: original[STAND_NAMES[0]][2:],
            STAND_NAMES[1]: original[STAND_NAMES[1]][2:],
            STAND_NAMES[2]: original[STAND_NAMES[2]][16:]}


def _copy_visible(surface, selection):
    """Copy portions of existing visible road/field geometry for CPU walking."""
    result = []
    for face, poly in zip(surface.faces, surface.polygons):
        for part in _pieces(poly.intersection(selection)):
            for xyz in _triangulate(part):
                out = np.asarray([_weights(face[:, (0, 2)], *p) @ face for p in xyz])
                original_winding = _cross2(face[1, (0, 2)]-face[0, (0, 2)], face[2, (0, 2)]-face[0, (0, 2)])
                if _cross2(xyz[1]-xyz[0], xyz[2]-xyz[0]) * original_winding < 0:
                    out = out[[0, 2, 1]]
                result.append(out)
    return result


def _new_paving(terrain, road, field, area, offset, height_at=None):
    """Split on terrain faces so the new surface cannot cut through terrain."""
    result = []
    min_clearance = float("inf")
    for terrain_face, terrain_poly in zip(terrain.faces, terrain.polygons):
        for part in _pieces(terrain_poly.intersection(area)):
            for xz in _triangulate(part):
                rows = []
                for x, z in xz:
                    ground = float(_weights(terrain_face[:, (0, 2)], x, z) @ terrain_face[:, 1])
                    if height_at is not None and abs(float(height_at(x, z))-ground) > .05:
                        raise ValueError("Provided terrain callback disagrees with the rendered mesh")
                    road_y = road.height(x, z, required=False)
                    field_y = field.height(x, z, required=False)
                    y = road_y if road_y is not None else field_y if field_y is not None else ground + offset
                    min_clearance = min(min_clearance, y-ground)
                    if y < ground-.005:
                        raise ValueError(f"Visible seam lies below terrain at {(x, z)}")
                    rows.append([x, y, z])
                p = np.asarray(rows)
                normal = np.cross(p[1]-p[0], p[2]-p[0])
                if normal[1] < 0:
                    p = p[[0, 2, 1]]
                    normal = -normal
                length = np.linalg.norm(normal)
                if length < 1e-9:
                    continue
                normal /= length
                result.append(np.asarray([(*v, *normal, v[0]/2.4, v[2]/2.4) for v in p]))
    if not result:
        raise ValueError("No new entry paving generated")
    return result, min_clearance


def _terrain_wall(line, top_height, terrain, normal_hint):
    """Close a supported edge, splitting its foot on rendered terrain triangles."""
    result = []
    seen = set()
    for terrain_poly in terrain.polygons:
        for section in _line_parts(line.intersection(terrain_poly)):
            if section.length < .001:
                continue
            a, b = np.asarray(section.coords[0]), np.asarray(section.coords[-1])
            key = tuple(sorted((tuple(np.round(a, 5)), tuple(np.round(b, 5)))))
            if key in seen:
                continue
            seen.add(key)
            ya, yb = top_height(*a), top_height(*b)
            ga, gb = terrain.height(*a)-.02, terrain.height(*b)-.02
            if min(ya-ga, yb-gb) < 0:
                raise ValueError("Stand wall top would be below its terrain")
            points = [np.array((a[0], ga, a[1])), np.array((b[0], gb, b[1])),
                      np.array((b[0], yb, b[1])), np.array((a[0], ya, a[1]))]
            for ids in ((0, 1, 2), (0, 2, 3)):
                p = np.asarray([points[i] for i in ids])
                normal = np.cross(p[1]-p[0], p[2]-p[0])
                if np.dot(normal[[0, 2]], normal_hint) < 0:
                    p = p[[0, 2, 1]]
                    normal = -normal
                normal /= np.linalg.norm(normal)
                result.append(np.asarray([(*v, *normal, v[0]/2, v[1]/2) for v in p]))
    return result


def _entry_edges(original_base, frame, opening, terrain):
    """Close the two cut stand ends with estimated concrete faces."""
    result = []
    lo, hi = opening["along_m"]
    for side in (-opening["half_width_m"], opening["half_width_m"]):
        side_line = LineString((frame.xz(lo, side), frame.xz(hi, side)))
        for tri in original_base:
            poly = Polygon(tri[:, (0, 2)])
            if poly.area < 1e-8:
                continue
            for line in _line_parts(poly.intersection(side_line)):
                result.extend(_terrain_wall(line, lambda x, z: float(_weights(tri[:, (0, 2)], x, z) @ tri[:, 1]),
                                            terrain, -np.sign(side)*frame.across))
    if not result:
        raise ValueError("No stand-end faces generated")
    return result


def _face_height(face, x, z):
    return float(_weights(face[:, (0, 2)], x, z) @ face[:, 1])


def _supported_line_pieces(line, faces):
    """Split a facade edge wherever the highest retained stand face changes."""
    intervals, positions = [], [0.0, line.length]
    for face in faces:
        poly = Polygon(face[:, (0, 2)])
        if poly.area < 1e-8 or not poly.intersects(line):
            continue
        for part in _line_parts(poly.intersection(line)):
            if part.length < 1e-6:
                continue
            a = line.project(Point(part.coords[0])); b = line.project(Point(part.coords[-1]))
            lo, hi = sorted((a, b))
            intervals.append((lo, hi, face))
            positions.extend((lo, hi))
    # 相邻斜面相交时在交线上拆分，始终让墙顶接可见的最高保留面。
    for i, (a0, a1, face_a) in enumerate(intervals):
        for b0, b1, face_b in intervals[i+1:]:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi-lo < 1e-5:
                continue
            p, q = line.interpolate(lo), line.interpolate(hi)
            da = _face_height(face_a, p.x, p.y)-_face_height(face_b, p.x, p.y)
            db = _face_height(face_a, q.x, q.y)-_face_height(face_b, q.x, q.y)
            if da*db < -1e-10:
                positions.append(lo+(hi-lo)*abs(da)/(abs(da)+abs(db)))
    positions = sorted(set(round(float(s), 9) for s in positions))
    for lo, hi in zip(positions[:-1], positions[1:]):
        if hi-lo < .001:
            continue
        mid = (lo+hi)/2
        candidates = [face for a, b, face in intervals if a-1e-6 <= mid <= b+1e-6]
        if not candidates:
            continue  # 未被保留看台覆盖的部分不生成立面。
        p = line.interpolate(mid)
        face = max(candidates, key=lambda t: _face_height(t, p.x, p.y))
        yield LineString((line.interpolate(lo), line.interpolate(hi))), face


def _exterior_facades(original_base, retained_base, frame, opening, terrain, segment_ids):
    """Estimated local rear closure; every wall top meets a retained stand."""
    slot = frame.rectangle(*opening["along_m"], opening["half_width_m"])
    result, source_length, supported_length = [], 0.0, 0.0
    for segment in segment_ids:
        tri = original_base[2*segment]
        source = LineString((tri[0, (0, 2)], tri[1, (0, 2)]))
        direction = tri[1, (0, 2)]-tri[0, (0, 2)]
        outward = np.array((direction[1], -direction[0]))
        for line in _line_parts(source.difference(slot)):
            source_length += line.length
            for piece, face in _supported_line_pieces(line, retained_base):
                supported_length += piece.length
                result.extend(_terrain_wall(piece, lambda x, z: _face_height(face, x, z), terrain, outward))
    if not result:
        raise ValueError("No supported stadium rear faces generated")
    return result, max(0.0, source_length-supported_length)


def _mesh_height(faces, x, z):
    hits = []
    for face in faces:
        if not (face[:,0].min()-1e-5 <= x <= face[:,0].max()+1e-5
                and face[:,2].min()-1e-5 <= z <= face[:,2].max()+1e-5):
            continue
        try:
            weights = _weights(face[:, (0, 2)], x, z)
        except ValueError:
            continue
        if min(weights) >= -1e-5:
            hits.append(float(weights @ face[:,1]))
    return max(hits) if hits else None


def _pole_support(mast_path, center, retained_stand):
    """Extend the original thin square pole down to its retained seat surface."""
    vertices = _read_mesh(mast_path).reshape(-1, 8)
    x, z = center
    near = vertices[(np.abs(vertices[:,0]-x) < .17) & (np.abs(vertices[:,2]-z) < .17)]
    if len(near) != 24:
        raise ValueError("Cannot identify the original northwest light pole")
    xmin, xmax = float(near[:,0].min()), float(near[:,0].max())
    zmin, zmax = float(near[:,2].min()), float(near[:,2].max())
    top = float(near[:,1].min())
    if (abs((xmax-xmin)-.32) > .002 or abs((zmax-zmin)-.32) > .002
            or float(near[:,1].max())-top < 20):
        raise ValueError("Unexpected light pole geometry")
    corners = ((xmin,zmin), (xmax,zmin), (xmax,zmax), (xmin,zmax))
    bottom = []
    for px, pz in corners:
        y = _mesh_height(retained_stand, px, pz)
        if y is None or y >= top-.1:
            raise ValueError("Light pole has no retained seat surface below its foot")
        bottom.append(y-.015)
    result = []
    for i in range(4):
        j = (i+1) % 4
        a, b = corners[i], corners[j]
        points = [np.array((a[0],bottom[i],a[1])), np.array((b[0],bottom[j],b[1])),
                  np.array((b[0],top,b[1])), np.array((a[0],top,a[1]))]
        for ids in ((0,1,2), (0,2,3)):
            p = np.asarray([points[k] for k in ids]); normal = np.cross(p[1]-p[0], p[2]-p[0])
            outward = (np.asarray(a)+np.asarray(b))/2-np.asarray(center)
            if np.dot(normal[[0, 2]], outward) < 0:
                p = p[[0, 2, 1]]
                normal = -normal
            normal /= np.linalg.norm(normal)
            result.append(np.asarray([(*v, *normal, v[1]/4, 0) for v in p]))
    return result


def _texture(path):
    # 仅表达照片中的红褐铺装色块；砖尺寸与材质并未得到照片证实。
    im = Image.new("RGB", (256, 256), (153, 98, 78))
    draw = ImageDraw.Draw(im)
    for row in range(8):
        for col in range(-1, 5):
            x = col*64 + (32 if row % 2 else 0)
            y = row*32
            variation = ((row*17 + col*11) % 9) - 4
            draw.rectangle((x+1, y+1, x+62, y+30), fill=(157+variation, 102+variation, 81+variation))
    im.save(path)


def _projected_geometry(triangles, selection):
    parts = []
    for tri in triangles:
        xz = tri[:, (0, 2)]
        if (xz[:,0].max() < selection.bounds[0] or xz[:,0].min() > selection.bounds[2]
                or xz[:,1].max() < selection.bounds[1] or xz[:,1].min() > selection.bounds[3]):
            continue
        poly = Polygon(xz)
        part = poly if poly.area > 1e-8 else LineString(xz)
        if part.intersects(selection):
            parts.append(part)
    return unary_union(parts)


def _routes(frame, cfg, visible_walk, terrain, obstacles):
    # 模拟 C++ 对踏面的最高可达选择；起点读取现有车道的真实上表面。
    result, tsv = {}, []
    spec = cfg["routes"]
    start, end = spec["along_m"]
    count = math.ceil((end-start)/spec["sample_step_m"])
    worst_step, min_margin = 0.0, float("inf")
    for name, offset in zip(("right", "center", "left"), spec["offsets_m"]):
        rows = []
        for distance in np.linspace(start, end, count+1):
            x, z = frame.xz(distance, offset)
            hits = []
            for face in visible_walk:
                w = _weights(face[:, (0, 2)], x, z)
                if min(w) >= -1e-5 and min(face[:, 4]) >= .7:
                    hits.append(float(w @ face[:, 1]))
            if not hits:
                raise ValueError(f"Visible walking surface is missing at {(x, z)}")
            y = max(hits)
            if y < terrain.height(x, z)-.02:
                raise ValueError("Visible walk surface is below terrain")
            if rows:
                worst_step = max(worst_step, abs(y-rows[-1][1]))
                if abs(y-rows[-1][1]) > .31:
                    raise ValueError(f"Entry route jumps more than .31 m at {(x, z)}")
            rows.append((float(x), y, float(z)))
            tsv.append(f"P\t{name}\t{x:.5f}\t{y:.5f}\t{z:.5f}")
        line = LineString([(x, z) for x, _, z in rows])
        min_margin = min(min_margin, line.distance(obstacles))
        if min_margin < .35:
            raise ValueError(f"Stand or edge geometry blocks the {name} route")
        result[name] = [[round(x, 5), round(y, 5), round(z, 5)] for x, y, z in rows]
    return result, tsv, worst_step, min_margin


def patch_assets(output_assets, base_assets, height_at=None, frame=None):
    """Patch originals into output_assets; return counts and validation data.

    Optional height_at must be the actual rendered terrain sampler.  Optional
    frame uses the same origin/inside_direction keys as the JSON.  Inputs are
    all read before writing, permitting a future generator to pass its own
    freshly generated directory as both base_assets and output_assets.
    """
    output_assets, base_assets = Path(output_assets), Path(base_assets)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = Frame(frame or cfg["frame"])
    original = {name: _read_mesh(base_assets/name) for name in STAND_NAMES}
    fold_spec = cfg["fold_repair"]
    repaired = _remove_foldback(original, fold_spec)
    input_paths = {f"baseline/{name}": base_assets/name for name in STAND_NAMES}
    input_paths.update({"terrain": _terrain_path, "road": _road_path, "field": _field_path})
    inputs = {label: _hash(path) for label, path in input_paths.items()}
    protected_paths = {name: p for name in PROTECTED_NAMES if (p := _field_path.parent/name).exists()}
    protected = {name: _hash(path) for name, path in protected_paths.items()}
    open_spec, pave_spec = cfg["opening"], cfg["paving"]
    pave_area = frame.rectangle(*pave_spec["along_m"], pave_spec["half_width_m"])
    selection = frame.rectangle(-8, 18, 7)
    outer_edges = [LineString((original[STAND_NAMES[0]][2*i, 0, (0, 2)],
                               original[STAND_NAMES[0]][2*i, 1, (0, 2)]))
                   for i in fold_spec["facade_segments"]]
    terrain = Surface(_terrain_path, unary_union([selection] + [line.buffer(1) for line in outer_edges]))
    road = Surface(_road_path, selection)
    field = Surface(_field_path, selection)
    gap = pave_area.difference(road.domain.union(field.domain))
    if gap.is_empty or not 20 < gap.area < 65:
        raise ValueError(f"Unexpected paving gap: {gap.area:.3f} m²")
    cut, changed = {}, {}
    for name in STAND_NAMES:
        cut[name], changed[name] = _cut_stands(repaired[name], frame, open_spec)
    paving, clearance = _new_paving(terrain, road, field, gap, pave_spec["terrain_offset_m"], height_at)
    paving_domain = unary_union([Polygon(t[:, (0, 2)]) for t in paving])
    coverage_error = paving_domain.symmetric_difference(gap).area
    road_overlap = paving_domain.intersection(road.domain).area
    field_overlap = paving_domain.intersection(field.domain).area
    if max(coverage_error, road_overlap, field_overlap) > .001:
        raise ValueError("Entry paving has a projected gap or overlaps an existing surface")
    edges = _entry_edges(repaired[STAND_NAMES[0]], frame, open_spec, terrain)
    rear, unsupported = _exterior_facades(original[STAND_NAMES[0]], cut[STAND_NAMES[0]],
                                           frame, open_spec, terrain, fold_spec["facade_segments"])
    edges.extend(rear)
    metal = _pole_support(_field_path.parent/"campus-stadium-masts.mesh", fold_spec["light_pole_xz_m"],
                          cut[STAND_NAMES[0]] + cut[STAND_NAMES[1]])
    visible_walk = paving + _copy_visible(road, pave_area) + _copy_visible(field, pave_area)
    obstacle_geometry = _projected_geometry([t for name in STAND_NAMES for t in cut[name]] + edges + metal, selection)
    route_spec = cfg["routes"]
    for side in route_spec["offsets_m"]:
        start = frame.xz(route_spec["along_m"][0], side)
        end = frame.xz(route_spec["along_m"][1], side)
        if road.height(*start, required=False) is None or field.height(*end, required=False) is None:
            raise ValueError("A route must start on the visible road and end on the visible field")
    routes, tsv, worst, margin = _routes(frame, cfg, visible_walk, terrain, obstacle_geometry)
    for label, path in input_paths.items():
        if _hash(path) != inputs[label]:
            raise ValueError(f"Read-only input changed during generation: {path}")
    output_assets.mkdir(parents=True, exist_ok=True)
    counts = {name: _save_mesh(output_assets/name, cut[name]) for name in STAND_NAMES}
    for name, values in (("campus-stadium-entry-paving.mesh", paving),
                         ("campus-stadium-entry-edges.mesh", edges),
                         ("campus-stadium-entry-metal.mesh", metal),
                         ("campus-stadium-entry-walk.mesh", visible_walk)):
        counts[name] = _save_mesh(output_assets/name, values)
    _texture(output_assets/"stadium-entry-paving.ppm")
    for name, path in protected_paths.items():
        if _hash(path) != protected[name]:
            raise ValueError(f"Untouched stadium resource changed: {path}")
    report = {"description": cfg["description"], "sources": cfg["sources"],
              "fold_repair": fold_spec["description"],
              "unsupported_facade_length_not_built_m": round(unsupported, 6),
              "frame": {"origin_xz_m": frame.origin.tolist(), "inside_unit_xz": frame.tangent.tolist(),
                        "across_unit_xz": frame.across.tolist()},
              "opening": open_spec,
              "opening_boundary_xz_m": [list(p) for p in frame.rectangle(
                  *open_spec["along_m"], open_spec["half_width_m"]).exterior.coords],
              "paving": pave_spec,
              "changed_source_triangles": changed, "input_sha256": inputs,
              "untouched_stadium_sha256": protected,
              "new_paving_area_m2": round(gap.area, 6),
              "paving_coverage_error_m2": round(coverage_error, 8),
              "paving_road_overlap_m2": round(road_overlap, 8),
              "paving_field_overlap_m2": round(field_overlap, 8),
              "min_paving_clearance_above_terrain_m": round(clearance, 6),
              "max_sampled_route_step_m": round(worst, 6),
              "min_projected_route_clearance_to_stand_m": round(margin, 6),
              "route_coordinates": "world x, expected visible foot y, world z",
              "routes": routes}
    (output_assets/"campus-stadium-entry-routes.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output_assets/"campus-stadium-entry-routes.tsv").write_text("\n".join(tsv)+"\n", encoding="utf-8")
    # 与 make_stadium 的返回格式一致，供全校园生成器直接更新资源计数。
    return counts


def main():
    global _terrain_path, _road_path, _field_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--terrain", type=Path, default=_terrain_path)
    parser.add_argument("--road", type=Path, default=_road_path)
    parser.add_argument("--field", type=Path, default=_field_path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _terrain_path, _road_path, _field_path = args.terrain, args.road, args.field
    print(json.dumps(patch_assets(args.output, args.baseline), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
