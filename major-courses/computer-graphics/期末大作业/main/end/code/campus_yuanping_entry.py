"""Add estimated Yuanping exterior stairs and continuous entry surfaces.

The original campus generator owns the frozen building meshes and terrain.  This
module only patches those meshes, never imports that generator, and can also be
called after it has saved the originals.  Dimensions are visual estimates; see
data/yuanping-entry.json and data/yuanping-entry-sources.md.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import mapbox_earcut
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "data" / "yuanping-entry.json"
DEFAULT_BASE = HERE / "data" / "yuanping-entry-baseline"
DEFAULT_TERRAIN = HERE / "assets" / "campus-terrain.mesh"
DEFAULT_OUTPUT = HERE.parents[2] / "build" / "yuanping-entry-candidate" / "assets"
_terrain_path = DEFAULT_TERRAIN


def _read_mesh(path):
    data = Path(path).read_bytes()
    if data[:4] != b"SZM1" or len(data) < 8:
        raise ValueError(f"Invalid mesh: {path}")
    count, = struct.unpack_from("<I", data, 4)
    if count < 3 or count % 3 or len(data) != 8 + count * 32:
        raise ValueError(f"Invalid mesh length: {path}")
    return np.frombuffer(data, dtype="<f4", offset=8).reshape(-1, 3, 8).copy()


def _save_mesh(path, triangles):
    triangles = np.asarray(triangles, dtype="<f4").reshape(-1, 3, 8)
    count = len(triangles) * 3
    if count < 3 or count > 500000 or not np.isfinite(triangles).all():
        raise ValueError(f"Invalid generated mesh {path}: {count} vertices")
    Path(path).write_bytes(b"SZM1" + struct.pack("<I", count) + triangles.tobytes())
    return count


class Frame:
    def __init__(self, values):
        self.mid = np.asarray(values["mid"], dtype=float)
        self.t = np.asarray(values["tangent"], dtype=float)
        self.n = np.asarray(values["outward"], dtype=float)
        self.level = float(values["level"])
        if (self.mid.shape != (2,) or self.t.shape != (2,) or self.n.shape != (2,)
                or abs(np.linalg.norm(self.t) - 1) > .001
                or abs(np.linalg.norm(self.n) - 1) > .001
                or abs(np.dot(self.t, self.n)) > .001):
            raise ValueError("Invalid Yuanping local coordinate frame")

    def xz(self, along, depth):
        return self.mid + along * self.t + depth * self.n

    def point(self, along, depth, y):
        x, z = self.xz(along, depth)
        return np.array((x, y, z), dtype=float)

    def local(self, xz):
        p = np.asarray(xz, dtype=float) - self.mid
        return np.stack((p @ self.t, p @ self.n), axis=-1)


class Mesh:
    def __init__(self, triangles=()):
        self.tris = [np.asarray(t, dtype=float) for t in triangles]

    def triangle(self, points, normal, uvs):
        self.tris.append(np.asarray([(*p, *normal, *uv) for p, uv in zip(points, uvs)], dtype=float))

    def quad(self, a, b, c, d, normal, uv=None):
        uv = uv or ((0, 0), (1, 0), (1, 1), (0, 1))
        self.triangle((a, b, c), normal, uv[:3])
        self.triangle((a, c, d), normal, (uv[0], uv[2], uv[3]))

    def horizontal(self, frame, s0, s1, d0, d1, y):
        # Normals are explicit, as in the existing campus mesh writer.
        self.quad(frame.point(s0, d0, y), frame.point(s1, d0, y),
                  frame.point(s1, d1, y), frame.point(s0, d1, y), (0, 1, 0))

    def beam(self, a, b, radius):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        delta = b - a
        length = np.linalg.norm(delta)
        if length < .01:
            return
        direction = delta / length
        ref = np.array((0., 1., 0.)) if abs(direction[1]) < .95 else np.array((1., 0., 0.))
        u = np.cross(direction, ref)
        u /= np.linalg.norm(u)
        v = np.cross(direction, u)
        ca = [a + radius * (su*u + sv*v) for su, sv in ((-1,-1), (1,-1), (1,1), (-1,1))]
        cb = [b + radius * (su*u + sv*v) for su, sv in ((-1,-1), (1,-1), (1,1), (-1,1))]
        for i in range(4):
            j = (i+1) % 4
            normal = np.cross(cb[i]-ca[i], ca[j]-ca[i])
            normal /= max(np.linalg.norm(normal), .0001)
            self.quad(ca[i], cb[i], cb[j], ca[j], normal)


def _pieces(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _pieces(part)


def _earcut(poly):
    rings = [list(poly.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in poly.interiors]
    rings = [ring for ring in rings if len(ring) >= 3]
    if not rings:
        return []
    vertices = np.asarray([p for ring in rings for p in ring], dtype=np.float64)
    ends = np.cumsum([len(ring) for ring in rings], dtype=np.uint32)
    indices = mapbox_earcut.triangulate_float64(vertices, ends)
    return vertices[indices].reshape(-1, 3, 2)


def _weights(tri, x, y):
    a, b, c = tri
    den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
    if abs(den) < 1e-12:
        raise ValueError("Degenerate source triangle")
    u = ((b[1]-c[1])*(x-c[0]) + (c[0]-b[0])*(y-c[1])) / den
    v = ((c[1]-a[1])*(x-c[0]) + (a[0]-c[0])*(y-c[1])) / den
    return np.array((u, v, 1-u-v))


class Terrain:
    """Sample the actual rendered terrain triangles, not the coarser DEM."""
    def __init__(self, path, frame):
        all_triangles = _read_mesh(path)
        local = frame.local(all_triangles[:, :, (0, 2)])
        mask = ((local[:, :, 0].max(axis=1) >= -52) & (local[:, :, 0].min(axis=1) <= 52)
                & (local[:, :, 1].max(axis=1) >= -5) & (local[:, :, 1].min(axis=1) <= 15))
        self.triangles = all_triangles[mask]
        self.local = local[mask]
        self.frame = frame
        self.cells = {}
        for i, tri in enumerate(self.local):
            lo = np.floor(tri.min(axis=0)/2).astype(int)
            hi = np.floor(tri.max(axis=0)/2).astype(int)
            for a in range(lo[0], hi[0]+1):
                for d in range(lo[1], hi[1]+1):
                    self.cells.setdefault((a,d), []).append(i)
        if not len(self.triangles):
            raise ValueError("No terrain triangles near Yuanping")

    def __call__(self, x, z):
        s, d = self.frame.local((x, z))
        ids = self.cells.get((math.floor(s/2), math.floor(d/2)), ())
        values = []
        for i in ids:
            try:
                w = _weights(self.local[i], s, d)
            except ValueError:
                continue
            if min(w) >= -1e-5:
                values.append(float(w @ self.triangles[i,:,1]))
        if not values:
            raise ValueError(f"No rendered terrain at {(x,z)} / {(s,d)}")
        if max(values)-min(values) > .03:
            raise ValueError(f"Overlapping terrain heights at {(x,z)}: {values}")
        return max(values)


def _cut_wing_doors(triangles, frame, door_s, door_y):
    result, changed = [], set()
    for tri in triangles:
        local = frame.local(tri[:, (0,2)])
        s, d = local[:,0], local[:,1]
        normal_out = float(np.dot(tri[0,(3,5)], frame.n))
        is_wing = (normal_out > .98 and 3.65 < float(np.mean(d)) < 3.95
                   and float(np.ptp(d)) < .2 and (s.min() > 28.5 or s.max() < -28.5))
        if not is_wing:
            result.append(tri)
            continue
        sy = np.column_stack((s, tri[:,1]))
        original = Polygon(sy)
        sign = 1 if s.mean() > 0 else -1
        left, right = sorted((sign*door_s[0], sign*door_s[1]))
        opening = box(left, *door_y[:1], right, door_y[1])
        clipped = original.difference(opening)
        if abs(clipped.area-original.area) < 1e-8:
            result.append(tri)
            continue
        changed.add(sign)
        winding = lambda a,b: a[0]*b[1]-a[1]*b[0]
        old_winding = winding(sy[1]-sy[0],sy[2]-sy[0])
        for poly in _pieces(clipped):
            for out in _earcut(poly):
                if winding(out[1]-out[0],out[2]-out[0])*old_winding < 0:
                    out = out[[0,2,1]]
                result.append(np.asarray([_weights(sy, *p) @ tri for p in out]))
    if changed != {-1,1}:
        raise ValueError(f"Expected door opening in both wing walls; found sides {changed}")
    return Mesh(result)


def _wing_plane(triangles, frame, sign, s):
    for tri in triangles:
        local = frame.local(tri[:,(0,2)])
        along, depth = local[:,0], local[:,1]
        if (float(np.dot(tri[0,(3,5)], frame.n)) > .98
                and 3.65 < float(np.mean(depth)) < 3.95
                and sign*along.mean() > 28.5 and along.min() <= sign*s <= along.max()):
            i, j = np.argmin(along), np.argmax(along)
            return float(depth[i] + (depth[j]-depth[i]) * (sign*s-along[i]) / (along[j]-along[i]))
    raise ValueError("Cannot find Yuanping wing wall")


def _keep_main_steps(triangles, frame):
    kept = []
    for tri in triangles:
        local = frame.local(tri[:,(0,2)])
        if np.max(np.abs(local[:,0])) < 10.1:
            kept.append(tri)
    if len(kept) != 28:
        raise ValueError(f"Unexpected baseline centre steps: {len(kept)} triangles")
    return Mesh(kept)


def _remove_old_platform_rails(triangles, frame):
    # The old mesh consists of 24-vertex beams. Remove only the 16 platform
    # railing beams; keep door mullions, roof trusses and all other details.
    vertices = triangles.reshape(-1,8)
    if len(vertices) % 24:
        raise ValueError("Unexpected baseline truss beam layout")
    out, removed = [], 0
    for beam in vertices.reshape(-1,8,3,8):
        v = beam.reshape(-1,8)
        local = frame.local(v[:,(0,2)])
        height = v[:,1]-frame.level
        is_rail = (local[:,1].min() > 2.85 and local[:,1].max() < 3.15
                   and height.min() > 3.05 and height.max() < 4.02
                   and np.min(np.abs(local[:,0])) > 18.8)
        if is_rail:
            removed += 1
        else:
            out.extend(beam)
    if removed != 16:
        raise ValueError(f"Expected 16 old platform rail beams; found {removed}")
    return Mesh(out)


def _wall_to_ground(mesh, frame, s0, d0, s1, d1, top0, top1, ground, normal):
    x0,z0 = frame.xz(s0,d0)
    x1,z1 = frame.xz(s1,d1)
    low0, low1 = min(top0, ground(x0,z0)-.08), min(top1, ground(x1,z1)-.08)
    for u in (.125,.25,.375,.5,.625,.75,.875):
        x,z=x0*(1-u)+x1*u,z0*(1-u)+z1*u
        if low0*(1-u)+low1*u > ground(x,z)+.01:
            raise ValueError(f"Foundation floats above terrain near {(x,z)}")
    mesh.quad(frame.point(s0,d0,low0),frame.point(s1,d1,low1),
              frame.point(s1,d1,top1),frame.point(s0,d0,top0),normal)


def _horizontal_pair(visible, walk, frame, s0, s1, d0, d1, y):
    visible.horizontal(frame,s0,s1,d0,d1,y)
    walk.horizontal(frame,s0,s1,d0,d1,y)


def _guard_line(mesh, frame, s0, d0, y0, s1, d1, y1, spacing=1.5):
    mesh.beam(frame.point(s0,d0,y0+.9), frame.point(s1,d1,y1+.9), .045)
    count = max(1, int(math.ceil(math.hypot(s1-s0,d1-d0)/spacing)))
    for i in range(count+1):
        f = i/count
        s, d, y = s0+(s1-s0)*f, d0+(d1-d0)*f, y0+(y1-y0)*f
        mesh.beam(frame.point(s,d,y+.05), frame.point(s,d,y+.9), .03)


def _sample_rect_max(ground, frame, s0, s1, d0, d1):
    ss = np.linspace(s0,s1,max(2,int(math.ceil(abs(s1-s0)/.20))+1))
    dd = np.linspace(d0,d1,max(2,int(math.ceil(abs(d1-d0)/.20))+1))
    return max(ground(*frame.xz(s,d)) for s in ss for d in dd)


def _add_entrance(steps, truss, shell, glass, walk, frame, cfg, base_tri, ground):
    central, spec = cfg["central_landing"], cfg["side_stairs"]
    h = frame.level + central["height"]
    w = central["half_width"]
    d0,d1 = central["depth"]
    door_w=central["door_half_width"]
    _horizontal_pair(steps,walk,frame,-w,w,0,d1,h)
    _horizontal_pair(steps,walk,frame,-door_w,door_w,d0,0,h)
    # The centre landing extends back to the recessed door; hide its vertical
    # edges against terrain instead of leaving an unsupported floating plane.
    for s in (-w,w):
        normal = (frame.t[0]*(1 if s>0 else -1),0,frame.t[1]*(1 if s>0 else -1))
        for a,b in zip(np.linspace(0,d1,9)[:-1],np.linspace(0,d1,9)[1:]):
            _wall_to_ground(steps,frame,s,a,s,b,h,h,ground,normal)
        for k in range(7):
            # Preserve the old shallow staircase while grounding both open
            # edges of every retained tread.
            a,b=d1+k*.54,d1+(k+1)*.54
            y=h-k*.15
            _wall_to_ground(steps,frame,s,a,s,b,y,y,ground,normal)
    for s in (-door_w,door_w):
        normal=(frame.t[0]*(1 if s>0 else -1),0,frame.t[1]*(1 if s>0 else -1))
        _wall_to_ground(steps,frame,s,d0,s,0,h,h,ground,normal)
    # Complete the short foundation below the lowest, outward-facing riser.
    s_grid=np.linspace(-w,w,81)
    for a,b in zip(s_grid[:-1],s_grid[1:]):
        _wall_to_ground(steps,frame,a,d1+7*.54,b,d1+7*.54,h-7*.15,h-7*.15,
                        ground,(frame.n[0],0,frame.n[1]))

    # The existing shallow front steps are kept. Add their upward faces only to
    # the walking mesh; walls and roofs never enter the walking model.
    for tri in steps.tris[:28]:
        if min(tri[:,4]) > .7:
            walk.tris.append(tri)

    bottoms, thresholds = {}, {}
    for sign in (-1,1):
        inner,outer = spec["inner"],spec["outer"]
        sd0,sd1 = spec["depth"]
        low_s = spec["landing_inner"]
        low = _sample_rect_max(ground,frame,sign*low_s,sign*inner,sd0,spec["landing_outer_depth"]) + spec["ground_clearance"]
        high = frame.level+spec["upper_height"]
        count = spec["count"]
        rise = (high-low)/count
        if not .08 <= rise <= .25:
            raise ValueError(f"Unexpected side stair rise: {rise}")
        bottoms[str(sign)] = low
        _horizontal_pair(steps,walk,frame,sign*low_s,sign*inner,sd0,spec["landing_outer_depth"],low)
        for d in (sd0,spec["landing_outer_depth"]):
            for a,b in zip(np.linspace(low_s,inner,5)[:-1],np.linspace(low_s,inner,5)[1:]):
                _wall_to_ground(steps,frame,sign*a,d,sign*b,d,low,low,ground,
                                (frame.n[0]*(1 if d>sd0 else -1),0,frame.n[1]*(1 if d>sd0 else -1)))
        for s in (low_s,inner):
            for a,b in zip(np.linspace(sd0,spec["landing_outer_depth"],17)[:-1],
                           np.linspace(sd0,spec["landing_outer_depth"],17)[1:]):
                normal=(sign*frame.t[0]*(1 if s==inner else -1),0,
                        sign*frame.t[1]*(1 if s==inner else -1))
                _wall_to_ground(steps,frame,sign*s,a,sign*s,b,low,low,ground,normal)
        for i in range(count):
            a,b = inner+(outer-inner)*i/count,inner+(outer-inner)*(i+1)/count
            y = low+(i+1)*rise
            _horizontal_pair(steps,walk,frame,sign*a,sign*b,sd0,sd1,y)
            n = (-sign*frame.t[0],0,-sign*frame.t[1])
            steps.quad(frame.point(sign*a,sd0,y-rise),frame.point(sign*a,sd1,y-rise),
                       frame.point(sign*a,sd1,y),frame.point(sign*a,sd0,y),n)
            for d in (sd0,sd1):
                normal = (frame.n[0]*(1 if d==sd1 else -1),0,frame.n[1]*(1 if d==sd1 else -1))
                _wall_to_ground(steps,frame,sign*a,d,sign*b,d,y,y,ground,normal)
        for d in (sd0,sd1):
            _guard_line(truss,frame,sign*inner,d,low,sign*outer,d,high)

        # High corridor is outside the wing wall. The tread ends at its own
        # landing, rather than rising through that corridor's underside.
        upper_end = spec["upper_end"]
        wall_d = _wing_plane(base_tri,frame,sign,(spec["door_range"][0]+spec["door_range"][1])/2)
        _horizontal_pair(steps,walk,frame,sign*outer,sign*upper_end,wall_d-.05,sd1,high)
        # Close the short gap between the landing and the inset glass, only
        # inside the estimated door opening. The common edge is exact, so the
        # threshold does not overlap the existing visible platform.
        door_a,door_b=spec["door_range"]
        _horizontal_pair(steps,walk,frame,sign*door_a,sign*door_b,wall_d-.20,wall_d-.05,high)
        thresholds[str(sign)]={"glass_depth":wall_d-.14,"floor_inner_depth":wall_d-.20,
                               "floor_outer_depth":wall_d-.05,"route_end_depth":wall_d-.10}
        _horizontal_pair(steps,walk,frame,sign*28.4,sign*outer,*spec["upper_walk_depth"],high)
        _horizontal_pair(steps,walk,frame,sign*26.5,sign*28.6,2.95,spec["upper_walk_depth"][1],high)
        # Give the elevated corridor a visible shallow fascia; the upper face
        # remains the sole walkable surface.
        for a,b in ((28.4,outer),(outer,upper_end)):
            d=spec["upper_walk_depth"][1] if a==28.4 else sd1
            steps.quad(frame.point(sign*a,d,high-.22),frame.point(sign*b,d,high-.22),
                       frame.point(sign*b,d,high),frame.point(sign*a,d,high),
                       (frame.n[0],0,frame.n[1]))
        # The original high platform is already rendered in shell; this copy is
        # deliberately walk-only, so the coplanar surfaces do not flicker.
        walk.horizontal(frame,sign*19,sign*28,.05,3.0,high)
        _guard_line(truss,frame,sign*19,3,high,sign*26.5,3,high)
        _guard_line(truss,frame,sign*29.2,spec["upper_walk_depth"][1],high,
                    sign*39.3,spec["upper_walk_depth"][1],high)
        _guard_line(truss,frame,sign*outer,sd1,high,sign*upper_end,sd1,high)
        _guard_line(truss,frame,sign*upper_end,wall_d+.2,high,sign*upper_end,sd1,high)

        # Estimated glazed upper opening; the façade itself was cut in both
        # old wall meshes. This describes an exterior threshold, not an interior.
        a,b = spec["door_range"]
        top = frame.level+spec["door_top"]
        door_d = wall_d-.14
        glass.quad(frame.point(sign*a,door_d,high),frame.point(sign*b,door_d,high),
                   frame.point(sign*b,door_d,top),frame.point(sign*a,door_d,top),
                   (frame.n[0],0,frame.n[1]))
        for s in (a,b):
            truss.beam(frame.point(sign*s,wall_d+.02,high),frame.point(sign*s,wall_d+.02,top),.045)
            shell.quad(frame.point(sign*s,door_d,high),frame.point(sign*s,wall_d+.01,high),
                       frame.point(sign*s,wall_d+.01,top),frame.point(sign*s,door_d,top),
                       (sign*frame.t[0],0,sign*frame.t[1]))
        truss.beam(frame.point(sign*a,wall_d+.02,top),frame.point(sign*b,wall_d+.02,top),.045)
        truss.beam(frame.point(sign*(a+b)/2,wall_d+.02,high),
                   frame.point(sign*(a+b)/2,wall_d+.02,top),.035)
    return bottoms,thresholds


def _paving_shape(cfg):
    p, spec = cfg["paving"], cfg["side_stairs"]
    shape = unary_union((box(-p["half_width"],p["depth"][0],p["half_width"],p["depth"][1]),
                         box(-10,6.78,10,p["depth"][0])))
    landings = [box(min(sign*spec["landing_inner"],sign*spec["inner"]),spec["depth"][0],
                    max(sign*spec["landing_inner"],sign*spec["inner"]),spec["landing_outer_depth"])
                for sign in (-1,1)]
    return shape.difference(unary_union(landings))


def _paving_triangles(frame, shape, ground, terrain, offset, sample_step=.25):
    if terrain is not None:
        source = zip(terrain.triangles,terrain.local)
    else:
        # Reusable generator entry point: sample a fine grid using the supplied
        # rendered-terrain callback. A small extra offset avoids interpolation
        # sinking into its piecewise-planar surface between grid vertices.
        lo,ld,hi,hd = shape.bounds
        step=sample_step
        ss=np.linspace(lo,hi,int(math.ceil((hi-lo)/step))+1)
        dd=np.linspace(ld,hd,int(math.ceil((hd-ld)/step))+1)
        grids=[]
        for a,b in zip(ss[:-1],ss[1:]):
            for c,d in zip(dd[:-1],dd[1:]):
                local=np.array(((a,c),(b,c),(b,d),(a,d)))
                v=np.array([(*frame.point(s,t,ground(*frame.xz(s,t))+.025),0,1,0,0,0) for s,t in local])
                grids.extend(((v[[0,1,2]],local[[0,1,2]]),(v[[0,2,3]],local[[0,2,3]])))
        source=grids
    mesh=Mesh()
    for tri,local in source:
        if (local[:,0].max() < shape.bounds[0] or local[:,0].min() > shape.bounds[2]
                or local[:,1].max() < shape.bounds[1] or local[:,1].min() > shape.bounds[3]):
            continue
        footprint=Polygon(local)
        if footprint.area < 1e-10:
            continue
        for poly in _pieces(footprint.intersection(shape)):
            if poly.area < 1e-8:
                continue
            for out in _earcut(poly):
                vertices=[]
                for s,d in out:
                    y=float(_weights(local,s,d) @ tri[:,1])+offset
                    vertices.append((*frame.point(s,d,y),0,1,0,s/2.4,d/1.2))
                mesh.tris.append(np.asarray(vertices))
    if not mesh.tris:
        raise ValueError("No ground paving intersects the terrain")
    return mesh


def _paving_texture(path):
    # Procedural gray rectangular joints: no photograph is redistributed.
    size=256
    rgb=np.empty((size,size,3),dtype=np.uint8)
    colors=((174,174,170),(165,166,163),(181,180,174),(170,171,168))
    for y in range(size):
        row=y//64
        for x in range(size):
            xx=(x+(32 if row%2 else 0))%256
            col=xx//64
            rgb[y,x]=((113,115,113) if xx%64<2 or y%64<2 else colors[(col+row*3)%4])
    Path(path).write_bytes(f"P6\n{size} {size}\n255\n".encode("ascii")+rgb.tobytes())


def _walk_at(walk_triangles, ground, x, z, previous):
    tri=walk_triangles
    mask=((tri[:,:,0].min(axis=1)-.001<=x) & (tri[:,:,0].max(axis=1)+.001>=x)
          & (tri[:,:,2].min(axis=1)-.001<=z) & (tri[:,:,2].max(axis=1)+.001>=z))
    base=ground(x,z)
    found=[base] if base <= previous+.8 else []
    for item in tri[mask]:
        try:
            weight=_weights(item[:,(0,2)],x,z)
        except ValueError:
            continue
        if min(weight) >= -1e-4:
            y=float(weight @ item[:,1])
            if y <= previous+.8 and y >= base-.05:
                found.append(y)
    return max(found) if found else base


def _check_walk_ground(walk, ground):
    min_clearance=float("inf")
    for tri in walk.tris:
        if np.min(tri[:,4]) < .7:
            raise ValueError("A non-walkable face entered the Yuanping walk mesh")
        for i in range(7):
            for j in range(7-i):
                u,v=i/6,j/6
                x,y,z=(u*tri[0,:3]+v*tri[1,:3]+(1-u-v)*tri[2,:3])
                clearance=float(y-ground(x,z))
                min_clearance=min(min_clearance,clearance)
                if clearance < -.025:
                    raise ValueError(f"Walk face intersects terrain at {(x,z)} by {-clearance:.3f} m")
    return min_clearance


def _routes(frame,cfg,walk,ground,bottoms,thresholds):
    tri=np.asarray(walk.tris)
    high=frame.level+cfg["side_stairs"]["upper_height"]
    inner=cfg["side_stairs"]["inner"]
    outer=cfg["side_stairs"]["outer"]
    routes={"central_entry": [(0,10),(0,6.7),(0,3.05),(0,.2)]}
    for sign in (-1,1):
        routes[f"side_{sign:+d}_stairs_to_upper_door"]=[(sign*26.75,9.9),(sign*26.75,8.5),
            (sign*26.75,6.3),(sign*inner,6.3),(sign*outer,6.3),(sign*41,6.3),
            (sign*41,thresholds[str(sign)]["route_end_depth"])]
        routes[f"side_{sign:+d}_upper_to_original_platform"]=[(sign*41,4.4),(sign*outer,4.4),
            (sign*28.5,4.4),(sign*27.5,4.4),(sign*27.5,2.5)]
    result={}
    for name,points in routes.items():
        positions=[]
        for a,b in zip(points[:-1],points[1:]):
            length=math.dist(a,b)
            count=max(1,math.ceil(length/.10))
            for i in range(count):
                u=i/count
                positions.append((a[0]*(1-u)+b[0]*u,a[1]*(1-u)+b[1]*u))
        positions.append(points[-1])
        def follow(samples):
            first=frame.xz(*samples[0])
            prev=high if "upper_to" in name else ground(*first)
            rows=[]
            for s,d in samples:
                x,z=frame.xz(s,d)
                y=_walk_at(tri,ground,x,z,prev)
                if abs(y-prev) > .31:
                    raise ValueError(f"Walk route {name} jumps {y-prev:.3f} at {(s,d)}")
                rows.append([round(float(x),4),round(float(z),4),round(float(y),4)])
                prev=y
            return rows
        forward=follow(positions)
        # Descending must begin at the actual top, not on the terrain below it.
        prev=forward[-1][2]
        for s,d in reversed(positions):
            x,z=frame.xz(s,d)
            y=_walk_at(tri,ground,x,z,prev)
            if abs(y-prev) > .31:
                raise ValueError(f"Reverse route {name} jumps {y-prev:.3f} at {(s,d)}")
            prev=y
        target=high if "side_" in name else frame.level-.10
        if abs(forward[-1][2]-target) > .08:
            raise ValueError(f"Route {name} missed its landing")
        result[name]=forward
    return {"description":"World x,z and expected foot y, approximately 0.1 m samples; both directions checked.",
            "frame": {"mid":frame.mid.tolist(),"tangent":frame.t.tolist(),
                      "outward":frame.n.tolist(),"level":frame.level},
            "estimated_lower_landings":bottoms,"estimated_upper_thresholds":thresholds,"routes":result}


def patch_assets(output_assets, base_assets, height_at=None, frame=None):
    """Patch frozen originals into output_assets and return generated counts.

    height_at(x,z) is the rendered terrain height; frame is a mapping with
    mid, tangent, outward and level.  If no callback is supplied, read the
    rendered campus terrain mesh specified by the CLI or DEFAULT_TERRAIN.
    The original meshes are all read before any output is written. A generator
    may therefore use the same input/output directory immediately after saving
    its original building; calling twice without regenerating originals fails.
    """
    output_assets,base_assets=Path(output_assets),Path(base_assets)
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"))
    frame=Frame(frame or cfg["frame"])
    terrain=None if height_at is not None else Terrain(_terrain_path,frame)
    ground=height_at or terrain
    read=lambda name:_read_mesh(base_assets/f"campus-yuanping-{name}.mesh")
    base0,shell0,truss0,glass0,steps0=(read(name) for name in ("base","shell","truss","glass","steps"))
    spec=cfg["side_stairs"]
    door_y=(frame.level+spec["upper_height"],frame.level+spec["door_top"])
    base=_cut_wing_doors(base0,frame,spec["door_range"],door_y)
    shell=_cut_wing_doors(shell0,frame,spec["door_range"],door_y)
    truss=_remove_old_platform_rails(truss0,frame)
    glass=Mesh(glass0)
    steps=_keep_main_steps(steps0,frame)
    walk=Mesh()
    bottoms,thresholds=_add_entrance(steps,truss,shell,glass,walk,frame,cfg,base0,ground)
    paving_shape=_paving_shape(cfg)
    paving=_paving_triangles(frame,paving_shape,ground,terrain,cfg["paving"]["terrain_offset"])
    # A coarser, slightly raised CPU-only path retains the connection without
    # adding thousands of cosmetic tile triangles to every camera query.
    path_walk=_paving_triangles(frame,paving_shape,ground,None,cfg["paving"]["terrain_offset"],sample_step=1.0)
    walk.tris.extend(path_walk.tris)
    routes=_routes(frame,cfg,walk,ground,bottoms,thresholds)
    routes["validation"]={"minimum_sampled_walk_clearance_m":_check_walk_ground(walk,ground),
                          "foundation_edges_checked_against_terrain":True}
    output_assets.mkdir(parents=True,exist_ok=True)
    result={}
    for name,mesh in (("base",base),("shell",shell),("truss",truss),("glass",glass),("steps",steps),
                      ("entry-walk",walk),("entry-paving",paving)):
        result[name]=_save_mesh(output_assets/f"campus-yuanping-{name}.mesh",mesh.tris)
    _paving_texture(output_assets/"yuanping-entry-paving.ppm")
    (output_assets/"campus-yuanping-entry-routes.json").write_text(
        json.dumps(routes,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    names={"central_entry":"center", "side_+1_stairs_to_upper_door":"north_stairs",
           "side_-1_stairs_to_upper_door":"south_stairs", "side_+1_upper_to_original_platform":"north_upper",
           "side_-1_upper_to_original_platform":"south_upper"}
    tsv=[]
    for name,rows in routes["routes"].items():
        tsv.extend(f"P\t{names[name]}\t{x:.4f}\t{y:.4f}\t{z:.4f}" for x,z,y in rows)
    (output_assets/"campus-yuanping-entry-routes.tsv").write_text("\n".join(tsv)+"\n",encoding="utf-8")
    return result


def main():
    global _terrain_path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASE)
    parser.add_argument("--terrain",type=Path,default=DEFAULT_TERRAIN)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    _terrain_path=args.terrain
    print(json.dumps(patch_assets(args.output,args.baseline),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
