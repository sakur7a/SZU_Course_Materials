"""Generate the north-library entry from existing campus assets.

The terrain and road are read-only inputs.  All positions and construction
details are model estimates; ``data/north-library-entry-sources.md`` describes
the photographs and plans and their limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import mapbox_earcut
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BASELINE = DATA / "north-library-entry-baseline"
CONFIG = DATA / "north-library-entry.json"
FRAME = DATA / "north-library-frame.json"
DEFAULT_INPUT = HERE / "assets"
DEFAULT_OUTPUT = HERE.parents[2] / "build" / "north-library-entry-candidate" / "assets"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_mesh(path):
    data = Path(path).read_bytes()
    if len(data) < 8 or data[:4] != b"SZM1":
        raise ValueError(f"Invalid mesh: {path}")
    count, = struct.unpack_from("<I", data, 4)
    if count % 3 or len(data) != 8 + 32 * count:
        raise ValueError(f"Invalid mesh length: {path}")
    return np.frombuffer(data, dtype="<f4", offset=8).reshape(-1, 3, 8).copy()


def _save_mesh(path, triangles):
    triangles = np.asarray(triangles, dtype="<f4").reshape(-1, 3, 8)
    if not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError(f"Empty or invalid mesh: {path}")
    Path(path).write_bytes(b"SZM1" + struct.pack("<I", len(triangles) * 3) + triangles.tobytes())


class Frame:
    def __init__(self, spec):
        self.cx, self.cz = spec["center_xz"]
        angle = math.radians(spec["rotation_y_degrees"])
        self.c, self.s = math.cos(angle), math.sin(angle)

    def world(self, x, z):
        return self.cx + self.c*x + self.s*z, self.cz - self.s*x + self.c*z

    def local(self, x, z):
        x, z = x-self.cx, z-self.cz
        return self.c*x-self.s*z, self.s*x+self.c*z


class Terrain:
    """Same triangle interpolation and patch ordering as TerrainHeights."""
    def __init__(self, assets):
        def grid(data, offset):
            nx, nz, x0, x1, z0, z1 = struct.unpack_from("<II4f", data, offset)
            offset += 24
            if nx < 2 or nz < 2 or nx > 1024 or nz > 1024:
                raise ValueError("Invalid terrain dimensions")
            values = np.frombuffer(data, dtype="<f4", count=nx*nz, offset=offset).reshape(nz, nx).copy()
            return (nx, nz, x0, x1, z0, z1, values), offset+4*nx*nz

        data = (assets / "campus-heights.bin").read_bytes()
        self.base, offset = grid(data, 0)
        if offset != len(data):
            raise ValueError("Invalid terrain data length")
        data = (assets / "campus-terrain-patches.bin").read_bytes()
        if data[:4] != b"SZP1":
            raise ValueError("Invalid terrain patches")
        count, = struct.unpack_from("<I", data, 4)
        self.patches, offset = [], 8
        for _ in range(count):
            item, offset = grid(data, offset)
            self.patches.append(item)
        if offset != len(data):
            raise ValueError("Invalid terrain patch length")

    def __call__(self, x, z):
        item = self.base
        for patch in self.patches:
            if patch[2] <= x <= patch[3] and patch[4] <= z <= patch[5]:
                item = patch
                break
        nx, nz, x0, x1, z0, z1, values = item
        u = min(nx-1, max(0, (x-x0)/(x1-x0)*(nx-1)))
        v = min(nz-1, max(0, (z-z0)/(z1-z0)*(nz-1)))
        i, j = min(int(u), nx-2), min(int(v), nz-2)
        a, b = u-i, v-j
        p, q, r, s = map(float, (values[j, i], values[j, i+1], values[j+1, i], values[j+1, i+1]))
        return (1-b)*p+(b-a)*r+a*s if b >= a else (1-a)*p+b*s+(a-b)*q


class Mesh:
    def __init__(self):
        self.triangles = []

    def triangle(self, points, up=False, uv=None):
        points = np.asarray(points, dtype=float)
        normal = np.cross(points[1]-points[0], points[2]-points[0])
        size = np.linalg.norm(normal)
        if size < 1e-9:
            return
        if up and normal[1] < 0:
            points = points[[0, 2, 1]]
            if uv is not None:
                uv = np.asarray(uv)[[0, 2, 1]]
            normal = -normal
        normal /= size
        if uv is None:
            if abs(normal[1]) >= .7:
                uv = [(p[0]/2, p[2]/2) for p in points]
            elif abs(normal[0]) >= abs(normal[2]):
                uv = [(p[2]/2, p[1]/2) for p in points]
            else:
                uv = [(p[0]/2, p[1]/2) for p in points]
        self.triangles.append(np.asarray([(*p, *normal, *t) for p, t in zip(points, uv)], dtype="<f4"))

    def quad(self, a, b, c, d, up=False):
        self.triangle((a, b, c), up=up)
        self.triangle((a, c, d), up=up)


def _clip_half(poly, values, inside):
    out = []
    for i, current in enumerate(poly):
        previous, ca, pa = poly[i-1], values[i], values[i-1]
        ci, pi = (ca >= 0, pa >= 0) if inside else (ca <= 0, pa <= 0)
        if ci != pi:
            out.append(previous + (current-previous)*(pa/(pa-ca)))
        if ci:
            out.append(current)
    return out


def _subtract_rectangle(triangles, frame, bounds):
    """Subtract only a rectangle, retaining interpolated position/normal/UV.

    Faces wholly outside remain byte-identical and in the same relative order.
    """
    xmin, xmax, zmin, zmax = bounds
    # Mesh positions are float32; tolerate sub-millimetre roundoff so that a
    # second invocation does not repeatedly re-triangulate boundary faces.
    epsilon = .00002
    kept, changed = [], 0
    for triangle in triangles:
        remaining, outside = [[np.asarray(v, dtype=float) for v in triangle]], []
        for axis, limit, sign in ((0, xmin, 1), (0, xmax, -1), (1, zmin, 1), (1, zmax, -1)):
            next_remaining = []
            for poly in remaining:
                v = np.asarray(poly)
                x, z = frame.local(v[:, 0], v[:, 2])
                vals = sign * ((x if axis == 0 else z)-limit)
                if np.min(vals) >= -epsilon:
                    next_remaining.append(poly)
                elif np.max(vals) <= epsilon:
                    outside.append(poly)
                else:
                    cut_out = _clip_half(poly, vals, False)
                    cut_in = _clip_half(poly, vals, True)
                    if len(cut_out) >= 3:
                        outside.append(cut_out)
                    if len(cut_in) >= 3:
                        next_remaining.append(cut_in)
            remaining = next_remaining
            if not remaining:
                break
        if not remaining:
            kept.append(triangle)
            continue
        changed += 1
        for poly in outside:
            for i in range(1, len(poly)-1):
                a, b, c = poly[0], poly[i], poly[i+1]
                if np.linalg.norm(np.cross(b[:3]-a[:3], c[:3]-a[:3])) > 1e-8:
                    kept.append(np.asarray((a, b, c), dtype="<f4"))
    return np.asarray(kept, dtype="<f4").reshape(-1, 3, 8), changed


def _strip_old_bridge(input_path, output_path, frame):
    expected = json.loads((BASELINE / "sources.json").read_text(encoding="utf8"))["campus-bridge.mesh"]
    if _hash(BASELINE / "campus-bridge.mesh") != expected:
        raise ValueError("North-library bridge baseline does not match its recorded SHA-256")
    raw = Path(input_path).read_bytes()
    faces = _read_mesh(input_path)
    x, z = frame.local(faces[:, :, 0], faces[:, :, 2])
    inner = ((x.min(1) >= -7.5) & (x.max(1) <= 10.7) & (z.min(1) >= 29.70) & (z.max(1) <= 41.08))
    outer = ((x.min(1) >= -1.8) & (x.max(1) <= 5.1) & (z.min(1) >= 56.90) & (z.max(1) <= 70.75))
    hits = np.flatnonzero(inner | outer)
    if hashlib.sha256(raw).hexdigest() == expected:
        if not np.array_equal(hits, np.arange(4987, 5135)):
            raise ValueError("Unexpected legacy library triangle layout")
        # Only the file header and the exact 148 legacy faces are changed.
        start, end = 8 + 4987*96, 8 + 5135*96
        Path(output_path).write_bytes(raw[:4] + struct.pack("<I", len(faces)*3-444) + raw[8:start] + raw[end:])
        return {"mode": "strict-baseline-migration", "removed_triangles": 148}
    if len(hits):
        raise ValueError("Changed bridge still contains geometry in legacy library footprints")
    # The full generator has already omitted the old faces; do not patch it again.
    Path(output_path).write_bytes(raw)
    return {"mode": "already-clean", "removed_triangles": 0}


class Road:
    """Find the actual road edge and its elevation from its rendered mesh."""
    def __init__(self, path, frame, probe_z):
        faces = _read_mesh(path)
        x, z = frame.local(faces[:, :, 0], faces[:, :, 2])
        selection = ((x.max(1) >= -4) & (x.min(1) <= 8)
                     & (z.max(1) >= probe_z-5) & (z.min(1) <= probe_z+5)
                     & (faces[:, :, 4].min(1) >= .7))
        self.faces, self.x, self.z = faces[selection], x[selection], z[selection]
        self.probe_z = probe_z
        if not len(self.faces):
            raise ValueError("No road mesh triangles at north-library entry")

    def edge(self, across):
        intervals = []
        for face, xs, zs in zip(self.faces, self.x, self.z):
            hits = []
            for j in range(3):
                k = (j+1) % 3
                if (min(xs[j], xs[k])-1e-8 <= across <= max(xs[j], xs[k])+1e-8
                        and abs(xs[j]-xs[k]) > 1e-10):
                    t = (across-xs[j])/(xs[k]-xs[j])
                    hits.append((float(zs[j]+t*(zs[k]-zs[j])),
                                 float(face[j, 1]+t*(face[k, 1]-face[j, 1]))))
            if len(hits) >= 2:
                hits.sort()
                intervals.append((hits[0], hits[-1]))
        intervals.sort()
        groups = []
        for interval in intervals:
            if not groups or interval[0][0] > max(v[1][0] for v in groups[-1])+1e-5:
                groups.append([interval])
            else:
                groups[-1].append(interval)
        for group in groups:
            lo = min(group, key=lambda v: v[0][0])[0]
            hi = max(group, key=lambda v: v[1][0])[1]
            if lo[0]-1e-5 <= self.probe_z <= hi[0]+1e-5:
                return lo
        raise ValueError(f"Road edge missing at local x={across}")

    def inner_edge_profile(self, left, right):
        # A linear segment's extremum lies at an end: include mesh vertices as
        # well as evenly spaced points when determining the safe stair limit.
        xs = set(np.linspace(left, right, int(math.ceil((right-left)/.20))+1))
        xs.update(float(x) for x in self.x.flat if left < x < right)
        return sorted((float(x), *self.edge(float(x))) for x in xs)


def _texture(path):
    """A neutral illustrative paving pattern, not a surveyed brick layout."""
    image = Image.new("RGB", (256, 256), (178, 176, 169))
    draw = ImageDraw.Draw(image)
    for row in range(16):
        z = row*16
        draw.line((0, z, 255, z), fill=(144, 143, 138), width=1)
        offset = 16 if row % 2 else 0
        for x in range(-32+offset, 288, 32):
            color = 173 + ((row*17+x*7) % 11)
            draw.rectangle((x+1, z+1, x+31, z+15), fill=(color+3, color+2, color-3))
            draw.line((x, z, x, z+15), fill=(144, 143, 138), width=1)
    image.save(path, format="PPM")


def _point(frame, x, y, z):
    wx, wz = frame.world(x, z)
    return (wx, float(y), wz)


def _intersect_rectangle(triangles, frame, bounds):
    xmin, xmax, zmin, zmax = bounds
    result = []
    for triangle in triangles:
        polys = [[np.asarray(v, dtype=float) for v in triangle]]
        for axis, limit, sign in ((0, xmin, 1), (0, xmax, -1), (1, zmin, 1), (1, zmax, -1)):
            next_polys = []
            for poly in polys:
                values = np.asarray(poly)
                x, z = frame.local(values[:, 0], values[:, 2])
                side = sign * ((x if axis == 0 else z)-limit)
                clipped = _clip_half(poly, side, True)
                if len(clipped) >= 3:
                    next_polys.append(clipped)
            polys = next_polys
            if not polys:
                break
        for poly in polys:
            for i in range(1, len(poly)-1):
                a, b, c = poly[0], poly[i], poly[i+1]
                if np.linalg.norm(np.cross(b[:3]-a[:3], c[:3]-a[:3])) > 1e-8:
                    result.append(np.asarray((a, b, c), dtype="<f4"))
    return result


def _pieces(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from _pieces(child)


def _earcut(polygon):
    rings = [list(polygon.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in polygon.interiors]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return
    vertices = np.asarray([p for r in rings for p in r], dtype=np.float64)
    ends = np.cumsum([len(r) for r in rings], dtype=np.uint32)
    indices = mapbox_earcut.triangulate_float64(vertices, ends)
    for tri in vertices[indices].reshape(-1, 3, 2):
        a, b = tri[1]-tri[0], tri[2]-tri[0]
        if abs(a[0]*b[1]-a[1]*b[0]) > 1e-8:
            yield tri


def _drape(mesh, walk, terrain_faces, frame, ground, area, offset):
    """Subdivide on the actual terrain triangles; preserve height at edges."""
    if area.is_empty:
        return
    aminx, aminz, amaxx, amaxz = area.bounds
    for face in terrain_faces:
        lx, lz = frame.local(face[:, 0], face[:, 2])
        if max(lx) < aminx or min(lx) > amaxx or max(lz) < aminz or min(lz) > amaxz:
            continue
        triangle = Polygon(zip(lx, lz))
        if triangle.area < 1e-8:
            continue
        for piece in _pieces(area.intersection(triangle)):
            for points in _earcut(piece):
                xyz = []
                for x, z in points:
                    wx, wz = frame.world(x, z)
                    xyz.append((wx, ground(wx, wz)+offset(float(x), float(z)), wz))
                mesh.triangle(xyz, up=True)
                walk.triangle(xyz, up=True)


def _wall(mesh, frame, ground, points):
    """Close a raised edge down into existing terrain at <=0.25 m intervals."""
    max_depth = 0.0
    for a, b in zip(points[:-1], points[1:]):
        length = math.hypot(b[0]-a[0], b[2]-a[2])
        samples = max(1, int(math.ceil(length/.25)))
        for j in range(samples):
            t0, t1 = j/samples, (j+1)/samples
            p = np.asarray(a)*(1-t0)+np.asarray(b)*t0
            q = np.asarray(a)*(1-t1)+np.asarray(b)*t1
            pg = ground(*frame.world(p[0], p[2]))
            qg = ground(*frame.world(q[0], q[2]))
            if min(p[1]-pg, q[1]-qg) < -.03:
                raise ValueError("Entry edge intersects the surrounding terrain")
            max_depth = max(max_depth, p[1]-pg, q[1]-qg)
            mesh.quad(_point(frame, p[0], pg-.08, p[2]), _point(frame, q[0], qg-.08, q[2]),
                      _point(frame, q[0], q[1], q[2]), _point(frame, p[0], p[1], p[2]))
    return max_depth


def _face_height(face, x, z):
    a, b, c = face
    den = (b[2]-c[2])*(a[0]-c[0]) + (c[0]-b[0])*(a[2]-c[2])
    if abs(den) < 1e-9:
        return None
    u = ((b[2]-c[2])*(x-c[0]) + (c[0]-b[0])*(z-c[2]))/den
    v = ((c[2]-a[2])*(x-c[0]) + (a[0]-c[0])*(z-c[2]))/den
    if min(u, v, 1-u-v) < -1e-4:
        return None
    return float(u*a[1]+v*b[1]+(1-u-v)*c[1])


def make_assets(output_assets, input_assets, height_at=None):
    """Build entry assets; terrain and road remain unchanged.

    ``input_assets`` is the current generator/runtime asset directory.  A
    verified old bridge is migrated once; an already-clean bridge is retained
    byte-for-byte.  ``height_at`` may be the generator's rendered-terrain
    sampler; otherwise the exact runtime height files are used.
    """
    output_assets, input_assets = Path(output_assets), Path(input_assets)
    output_assets.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    fs = json.loads(FRAME.read_text(encoding="utf8"))
    frame, ground = Frame(fs), height_at or Terrain(input_assets)
    axis = float(fs["main_stair_axis_x"])
    slab_y = float(ground(*frame.world(0, 0))) + float(fs["first_floor_top"])
    slab_front = float(fs["first_floor_half_extent"])
    narrow = float(cfg["lower_half_width_m"])
    wide = float(cfg["upper_half_width_m"])
    fore_out, fore_in = float(cfg["forecourt_outer_z"]), float(cfg["forecourt_inner_z"])
    offset = float(cfg["existing_path_offset_m"])
    road = Road(input_assets / "campus-road-surface.mesh", frame, float(cfg["road_probe_local_z"]))
    profile = road.inner_edge_profile(axis-narrow, axis+narrow)
    lower_start = min(item[1] for item in profile)-float(cfg["road_clearance_m"])
    low_count, low_depth = int(cfg["lower_tread_count"]), float(cfg["lower_tread_depth_m"])
    lower_end = lower_start-low_count*low_depth
    if not slab_front < fore_in < fore_out < lower_end < lower_start:
        raise ValueError("Invalid entry geometry ordering")

    road_polys = []
    for xs, zs in zip(road.x, road.z):
        p = Polygon(zip(xs, zs))
        if p.area > 1e-8:
            road_polys.append(p)
    low_domain = box(axis-narrow, fore_out, axis+narrow, lower_start)
    road_overlap = low_domain.intersection(unary_union(road_polys)).area
    if road_overlap > 1e-7:
        raise ValueError(f"Entry would overlap the road: {road_overlap} m²")

    recorded = json.loads((BASELINE / "sources.json").read_text(encoding="utf8"))
    for name, digest in recorded.items():
        if _hash(BASELINE / name) != digest:
            raise ValueError(f"North-library baseline hash mismatch: {name}")
    original_bridge = input_assets / "campus-bridge.mesh"
    original_path = input_assets / "campus-network-paths.mesh"
    if _hash(original_bridge) == recorded["campus-bridge.mesh"] and _hash(original_path) != recorded["campus-network-paths.mesh"]:
        raise ValueError("Legacy bridge requires the matching verified path baseline")
    original_paths = _read_mesh(original_path)
    terrain_faces = _read_mesh(input_assets / "campus-terrain.mesh")
    migration = _strip_old_bridge(original_bridge, output_assets / "campus-bridge.mesh", frame)

    # Only the two new stair/approach footprints replace shared path triangles.
    path, outer_cut = _subtract_rectangle(original_paths, frame,
        (axis-narrow, axis+narrow, fore_out, lower_start))
    path, upper_cut = _subtract_rectangle(path, frame,
        (axis-wide, axis+wide, slab_front, fore_in))
    _save_mesh(output_assets / "campus-network-paths.mesh", path)

    steps, paving, edges, walk = Mesh(), Mesh(), Mesh(), Mesh()
    spacing = float(cfg["cross_section_spacing_m"])
    max_depth, minimum_clearance = 0.0, float("inf")

    # Three shallow, visible lower steps start behind the road projection.
    xs = np.linspace(axis-narrow, axis+narrow, int(math.ceil(2*narrow/spacing))+1)
    low_base = np.array([ground(*frame.world(float(x), lower_end)) for x in xs])
    low_rise = float(cfg["lower_riser_m"])
    for i in range(low_count):
        z0, z1 = lower_start-i*low_depth, lower_start-(i+1)*low_depth
        current, previous = low_base+low_rise*(i+1), low_base+low_rise*i
        for j, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
            top = [_point(frame, x0, current[j], z0), _point(frame, x1, current[j+1], z0),
                   _point(frame, x1, current[j+1], z1), _point(frame, x0, current[j], z1)]
            steps.quad(*top, up=True)
            walk.quad(*top, up=True)
            steps.quad(_point(frame, x0, previous[j], z0), _point(frame, x1, previous[j+1], z0), top[1], top[0])
        for x, y in ((xs[0], current[0]), (xs[-1], current[-1])):
            max_depth = max(max_depth, _wall(edges, frame, ground, [(x, y, z0), (x, y, z1)]))

    # A thin, terrain-following connector closes the grade to the existing
    # path; the final 0.055 m offset is its actual source mesh offset.
    connector = box(axis-narrow, fore_out, axis+narrow, lower_end)
    def connector_offset(x, z):
        return offset + (low_rise*low_count-offset)*(z-fore_out)/(lower_end-fore_out)
    _drape(paving, walk, terrain_faces, frame, ground, connector, connector_offset)
    for x in (axis-narrow, axis+narrow):
        zs = np.linspace(fore_out, lower_end, int(math.ceil((lower_end-fore_out)/.25))+1)
        side = [(x, ground(*frame.world(x, float(z)))+connector_offset(x, float(z)), float(z)) for z in zs]
        max_depth = max(max_depth, _wall(edges, frame, ground, side))

    # Retain the original visible network path between the flights.  Copy only
    # its clipped top surfaces to the dedicated, invisible walk mesh.
    middle_bounds = (axis-narrow, axis+narrow, fore_in, fore_out)
    mx, mz = frame.local(path[:, :, 0], path[:, :, 2])
    sel = ((mx.max(1) >= middle_bounds[0]) & (mx.min(1) <= middle_bounds[1])
           & (mz.max(1) >= middle_bounds[2]) & (mz.min(1) <= middle_bounds[3])
           & (path[:, :, 4].min(1) >= .7))
    path_walk = _intersect_rectangle(path[sel], frame, middle_bounds)
    walk.triangles.extend(path_walk)
    footprint = box(*[middle_bounds[0], middle_bounds[2], middle_bounds[1], middle_bounds[3]])
    covered = []
    for tri in path_walk:
        x, z = frame.local(tri[:, 0], tri[:, 2])
        poly = Polygon(zip(x, z))
        if poly.area > 1e-8:
            covered.append(poly)
    if not covered:
        raise ValueError("The existing forecourt path has no walkable surfaces")
    missing = footprint.difference(unary_union(covered))
    if missing.area > .5:
        raise ValueError(f"Unexpected gap in existing forecourt: {missing.area} m²")
    _drape(paving, walk, terrain_faces, frame, ground, missing, lambda x, z: offset)

    # The wider flight follows the actual cross-slope at its first edge and
    # ends exactly at the visible first-floor slab top.  It does not redraw
    # the slab; the full slab is included only in the invisible walk mesh.
    xs = np.linspace(axis-wide, axis+wide, int(math.ceil(2*wide/spacing))+1)
    bases = np.array([ground(*frame.world(float(x), fore_in))+offset for x in xs])
    upper_count = max(1, int(math.ceil(max(slab_y-bases)/float(cfg["upper_max_riser_m"]))))
    upper_depth = (fore_in-slab_front)/upper_count
    if min(slab_y-bases) <= 0:
        raise ValueError("Wide-stair start is not below the first-floor slab")
    for i in range(upper_count):
        z0, z1 = fore_in-i*upper_depth, fore_in-(i+1)*upper_depth
        current = bases+(slab_y-bases)*(i+1)/upper_count
        previous = bases+(slab_y-bases)*i/upper_count
        for j, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
            top = [_point(frame, x0, current[j], z0), _point(frame, x1, current[j+1], z0),
                   _point(frame, x1, current[j+1], z1), _point(frame, x0, current[j], z1)]
            steps.quad(*top, up=True)
            walk.quad(*top, up=True)
            steps.quad(_point(frame, x0, previous[j], z0), _point(frame, x1, previous[j+1], z0), top[1], top[0])
            for x in (float(x0), (float(x0)+float(x1))/2, float(x1)):
                y = float(np.interp(x, xs, current))
                for z in (z0, (z0+z1)/2, z1):
                    minimum_clearance = min(minimum_clearance, y-ground(*frame.world(x, z)))
        for x, y in ((xs[0], current[0]), (xs[-1], current[-1])):
            max_depth = max(max_depth, _wall(edges, frame, ground, [(x, y, z0), (x, y, z1)]))
    if minimum_clearance < .015:
        raise ValueError(f"Wide stair approaches/buries terrain: {minimum_clearance:.4f} m")
    h = float(fs["first_floor_half_extent"])
    walk.quad(_point(frame, -h, slab_y, h), _point(frame, h, slab_y, h),
              _point(frame, h, slab_y, -h), _point(frame, -h, slab_y, -h), up=True)

    _save_mesh(output_assets / "campus-north-library-entry-steps.mesh", steps.triangles)
    _save_mesh(output_assets / "campus-north-library-entry-paving.mesh", paving.triangles)
    _save_mesh(output_assets / "campus-north-library-entry-edges.mesh", edges.triangles)
    _save_mesh(output_assets / "campus-north-library-entry-walk.mesh", walk.triangles)
    _texture(output_assets / "north-library-entry-paving.ppm")
    routes = _routes(frame, ground, walk, road, cfg, axis, slab_y)
    (output_assets / "campus-north-library-entry-routes.tsv").write_text(
        "\n".join(f"P {route['id']} {p['x']:.8f} {p['y']:.8f} {p['z']:.8f}"
                  for route in routes for p in route["points"])+"\n", encoding="utf8")
    info = {"description": cfg["description"], "migration": migration,
            "slab_top_m": slab_y, "road_inner_edge_local_z_minmax": [min(p[1] for p in profile), max(p[1] for p in profile)],
            "road_overlap_m2": road_overlap, "lower_start_local_z": lower_start,
            "lower_treads": low_count, "lower_riser_m": low_rise,
            "upper_treads": upper_count, "upper_riser_minmax_m": [float(min((slab_y-bases)/upper_count)), float(max((slab_y-bases)/upper_count))],
            "upper_tread_depth_m": upper_depth, "upper_min_sampled_ground_clearance_m": minimum_clearance,
            "max_sampled_edge_height_m": max_depth, "forecourt_reused_path_m2": footprint.area-missing.area,
            "forecourt_repaired_gap_m2": missing.area, "shared_path_cut_triangles": [outer_cut, upper_cut],
            "routes": routes}
    (output_assets / "campus-north-library-entry-routes.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf8")
    return {k: v for k, v in info.items() if k != "routes"} | {"route_count": len(routes)}


def _routes(frame, ground, walk, road, cfg, center, slab_y):
    faces = np.asarray(walk.triangles, dtype=float)
    faces = faces[faces[:, :, 4].min(1) >= .7, :, :3]
    lo_x, hi_x = faces[:, :, 0].min(1), faces[:, :, 0].max(1)
    lo_z, hi_z = faces[:, :, 2].min(1), faces[:, :, 2].max(1)

    def surface(x, z, base, limit):
        possible = np.flatnonzero((lo_x-.001 <= x) & (hi_x+.001 >= x)
                                  & (lo_z-.001 <= z) & (hi_z+.001 >= z))
        best = base if base <= limit else None
        for index in possible:
            y = _face_height(faces[index], x, z)
            if y is not None and y <= limit and y >= base-.05 and (best is None or y > best):
                best = y
        return base if best is None else best

    routes = []
    for name, delta in zip(("left", "center", "right"), cfg["routes"]["offsets_m"]):
        local_x = center+float(delta)
        start = road.edge(local_x)[0]+.6
        end = float(cfg["routes"]["door_end_local_z"])
        n = int(math.ceil((start-end)/float(cfg["routes"]["sample_step_m"])))
        points = []
        for local_z in np.linspace(start, end, n+1):
            x, z = frame.world(local_x, float(local_z))
            g = float(ground(x, z))
            expected = surface(x, z, g, float("inf"))
            points.append({"x": float(x), "y": expected, "z": float(z)})
        if abs(points[-1]["y"]-slab_y) > .02:
            raise ValueError(f"Route {name} does not reach the first-floor slab")
        for reverse in (False, True):
            sequence = list(reversed(points)) if reverse else points
            previous = sequence[0]["y"]
            for point in sequence:
                g = float(ground(point["x"], point["z"]))
                actual = surface(point["x"], point["z"], g, previous+.8)
                if abs(actual-point["y"]) > .05 or abs(actual-previous) > .31:
                    raise ValueError(f"Discontinuous {name} route, reverse={reverse}: {point} got {actual}, previous {previous}")
                previous = actual
        max_step = max(abs(a["y"]-b["y"]) for a, b in zip(points, points[1:]))
        routes.append({"id": name, "local_x_m": local_x, "max_sample_step_m": max_step, "points": points})
    return routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(make_assets(args.output, args.input), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
