"""校门资源的独立几何检查；只用标准库，不启动生成器或图形窗口。

直接读取最终 SZM1 网格、地形、OSM、配置与运行时 TSV，检查可重复的几何
约束。车道和立心通廊的检查是采样验证，不表示实测门宽或现实开放状态。
运行：python main/end/code/verify_gate_assets.py --report build/gate-assets-check.json
"""
import argparse
import csv
import json
import math
import re
import struct
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
EXPECTED_IDS = {'liye', 'lide', 'lizhi', 'ligong', 'lixin', 'liyan',
                'zhiyuan', 'zhixin', 'zhihe', 'zhicheng', 'zhiheng'}
# 实际入口的 OSM 道路；不是与校门最近的任意公路。
ROAD_IDS = {
    'liye': ['231407871'], 'lide': ['220071398', '220071403'],
    'ligong': ['222088986'], 'liyan': ['231085451'],
    'zhiyuan': ['491931315'], 'zhixin': ['722200013'],
    'zhihe': ['1266340897', '1266340898'],
    'zhicheng': ['976045162', '220883467'], 'zhiheng': ['220883470'],
}


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def segment_triangle(start, end, tri):
    """有限线段与三角面相交，返回交点，排除三角面平面的无限延长部分。"""
    a, b, c = (v[:3] for v in tri)
    direction = sub(end, start)
    e1, e2 = sub(b, a), sub(c, a)
    h = cross(direction, e2)
    det = dot(e1, h)
    if abs(det) < 1e-10:
        return None
    inv = 1/det
    s = sub(start, a)
    u = dot(s, h)*inv
    q = cross(s, e1)
    v = dot(direction, q)*inv
    t = dot(e2, q)*inv
    if -.000001 <= u and -.000001 <= v and u+v <= 1.000001 and 0 <= t <= 1:
        return tuple(start[i]+t*direction[i] for i in range(3))
    return None


def surface_hits(vertices, x, z):
    """按水平投影重心坐标取向上表面高度，支持台阶与斜坡。"""
    found = []
    for i in range(0, len(vertices), 3):
        a, b, c = vertices[i:i+3]
        if min(a[4], b[4], c[4]) < .7:
            continue
        den = (b[2]-c[2])*(a[0]-c[0])+(c[0]-b[0])*(a[2]-c[2])
        if abs(den) < 1e-8:
            continue
        u = ((b[2]-c[2])*(x-c[0])+(c[0]-b[0])*(z-c[2]))/den
        v = ((c[2]-a[2])*(x-c[0])+(a[0]-c[0])*(z-c[2]))/den
        w = 1-u-v
        if min(u, v, w) >= -1e-4:
            found.append(u*a[1]+v*b[1]+w*c[1])
    return found


class Terrain:
    """使用与渲染三角面相同的插值，不能用双线性 DEM 代替。"""
    def __init__(self, assets):
        blob = (assets/'campus-heights.bin').read_bytes()
        self.coarse, offset = self.grid(blob, 0)
        if offset != len(blob):
            raise ValueError('Unexpected trailing coarse terrain data')
        blob = (assets/'campus-terrain-patches.bin').read_bytes()
        if blob[:4] != b'SZP1':
            raise ValueError('Invalid terrain patch header')
        count, = struct.unpack_from('<I', blob, 4)
        self.patches, offset = [], 8
        for _ in range(count):
            grid, offset = self.grid(blob, offset)
            self.patches.append(grid)
        if offset != len(blob):
            raise ValueError('Unexpected trailing terrain patch data')

    @staticmethod
    def grid(blob, offset):
        nx, nz, x0, x1, z0, z1 = struct.unpack_from('<II4f', blob, offset)
        if nx < 2 or nz < 2 or nx*nz > 20_000_000 or not x1 > x0 or not z1 > z0:
            raise ValueError('Invalid terrain grid dimensions')
        offset += 24
        values = struct.unpack_from('<'+str(nx*nz)+'f', blob, offset)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite terrain height')
        return (nx, nz, x0, x1, z0, z1, values), offset+nx*nz*4

    def __call__(self, x, z):
        grid = self.coarse
        for patch in self.patches:
            if patch[2] <= x <= patch[3] and patch[4] <= z <= patch[5]:
                grid = patch
                break
        nx, nz, x0, x1, z0, z1, h = grid
        u = max(0, min(nx-1, (x-x0)/(x1-x0)*(nx-1)))
        v = max(0, min(nz-1, (z-z0)/(z1-z0)*(nz-1)))
        i, j = min(int(u), nx-2), min(int(v), nz-2)
        a, b = u-i, v-j
        p, q, r, s = h[j*nx+i], h[j*nx+i+1], h[(j+1)*nx+i], h[(j+1)*nx+i+1]
        return (1-b)*p+(b-a)*r+a*s if b >= a else (1-a)*p+b*s+(a-b)*q


class Mesh:
    def __init__(self, path):
        raw = path.read_bytes()
        if raw[:4] != b'SZM1' or len(raw) < 8:
            raise ValueError(f'Invalid mesh header: {path.name}')
        count, = struct.unpack_from('<I', raw, 4)
        if count < 3 or count % 3 or len(raw) != 8+count*32:
            raise ValueError(f'Invalid vertex count or byte length: {path.name}')
        self.name = path.name
        self.vertices = list(struct.iter_unpack('<8f', raw[8:]))
        if not all(math.isfinite(x) for v in self.vertices for x in v):
            raise ValueError(f'Nonfinite vertex: {path.name}')
        if any(abs(math.sqrt(sum(v[i]*v[i] for i in (3, 4, 5)))-1) > .015
               for v in self.vertices):
            raise ValueError(f'Nonunit normal: {path.name}')
        self.bounds = [[min(v[i] for v in self.vertices) for i in range(3)],
                       [max(v[i] for v in self.vertices) for i in range(3)]]
        self.triangles = []
        for i in range(0, count, 3):
            tri = self.vertices[i:i+3]
            bounds = ([min(v[k] for v in tri) for k in range(3)],
                      [max(v[k] for v in tri) for k in range(3)])
            self.triangles.append((tri, bounds))

    def first_hit(self, a, b):
        lower = [min(x, y)-1e-5 for x, y in zip(a, b)]
        upper = [max(x, y)+1e-5 for x, y in zip(a, b)]
        for tri, bounds in self.triangles:
            if any(bounds[0][i] > upper[i] or bounds[1][i] < lower[i] for i in range(3)):
                continue
            point = segment_triangle(a, b, tri)
            if point is not None:
                return point
        return None


def point_in_ring(p, ring):
    x, z = p
    inside = False
    for a, b in zip(ring, ring[1:]+ring[:1]):
        if (a[1] > z) != (b[1] > z) and x < (b[0]-a[0])*(z-a[1])/(b[1]-a[1])+a[0]:
            inside = not inside
    return inside


def point_segment(p, a, b):
    ab = sub(b, a)
    t = max(0, min(1, dot(sub(p, a), ab)/max(1e-12, dot(ab, ab))))
    q = (a[0]+t*ab[0], a[1]+t*ab[1])
    return distance(p, q), t, q


class OSM:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.nodes = {n.attrib['id']: ((float(n.attrib['lon'])-113.9318)*102800,
                                      (22.5355-float(n.attrib['lat']))*111320)
                      for n in root.findall('node')}
        self.ways = {w.attrib['id']: [n.attrib['ref'] for n in w.findall('nd')]
                     for w in root.findall('way')}
        self.relations = {r.attrib['id']: [(m.attrib['ref'], m.attrib.get('role', ''))
                                        for m in r.findall('member') if m.attrib['type'] == 'way']
                          for r in root.findall('relation')}

    def line(self, ident):
        return [self.nodes[n] for n in self.ways[ident] if n in self.nodes]

    def rings(self, reference):
        kind, ident = reference.split('/')
        if kind == 'way':
            return [self.line(ident)] if ident in self.ways else []
        # 将多边形 relation 外环的 way 按相同端点接回；内环不当作楼体。
        parts = [self.ways[w][:] for w, role in self.relations.get(ident, [])
                 if role in ('', 'outer') and w in self.ways]
        rings = []
        while parts:
            ring = parts.pop()
            changed = True
            while changed and ring[0] != ring[-1]:
                changed = False
                for i, part in enumerate(parts):
                    if part[0] == ring[-1]:
                        ring.extend(part[1:])
                    elif part[-1] == ring[-1]:
                        ring.extend(part[-2::-1])
                    elif part[-1] == ring[0]:
                        ring = part[:-1]+ring
                    elif part[0] == ring[0]:
                        ring = part[:0:-1]+ring
                    else:
                        continue
                    parts.pop(i)
                    changed = True
                    break
            if ring[0] == ring[-1]:
                rings.append([self.nodes[n] for n in ring if n in self.nodes])
        return rings


def line_sample(line, along):
    lengths = [distance(a, b) for a, b in zip(line, line[1:])]
    remaining = max(0, min(sum(lengths), along))
    for a, b, length in zip(line, line[1:], lengths):
        if remaining <= length+1e-8:
            t = remaining/max(length, 1e-9)
            return (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])), ((b[0]-a[0])/length, (b[1]-a[1])/length)
        remaining -= length
    raise ValueError('Empty mapped road')


def closest_along(line, p):
    best = (float('inf'), 0)
    total = 0
    for a, b in zip(line, line[1:]):
        dist, t, _ = point_segment(p, a, b)
        length = distance(a, b)
        best = min(best, (dist, total+t*length))
        total += length
    return best


class Check:
    def __init__(self):
        self.errors, self.warnings, self.metrics = [], [], {}

    def error(self, gate, check, detail):
        self.errors.append({'gate': gate, 'check': check, 'detail': detail})

    def warning(self, gate, check, detail):
        self.warnings.append({'gate': gate, 'check': check, 'detail': detail})


def validate_ppm(path):
    raw = path.read_bytes()
    match = re.match(rb'P6\s+(\d+)\s+(\d+)\s+(\d+)\s', raw)
    if not match or int(match[3]) != 255:
        raise ValueError(f'Invalid generated P6 image: {path.name}')
    width, height = int(match[1]), int(match[2])
    if min(width, height) < 128 or len(raw)-match.end() != width*height*3:
        raise ValueError(f'Truncated or implausible sign texture: {path.name}')


def validate_manifests(assets, data, result):
    gates = json.loads((assets/'campus-gates.json').read_text(encoding='utf-8'))
    configs = {}
    for filename in ('gates-west.json', 'gates-north.json', 'gates-south.json'):
        for gate in json.loads((data/filename).read_text(encoding='utf-8'))['gates']:
            if gate['id'] in configs:
                raise ValueError('Duplicate configured gate '+gate['id'])
            configs[gate['id']] = gate
    by_id = {g['id']: g for g in gates}
    if len(by_id) != len(gates) or set(by_id) != EXPECTED_IDS or set(configs) != EXPECTED_IDS:
        raise ValueError('Gate IDs are duplicated, missing, or unexpected')
    rows = [line.split('\t') for line in (assets/'campus-gates.tsv').read_text(encoding='utf-8').splitlines()
            if line and not line.startswith('#')]
    if any(row[0] not in ('G', 'L', 'W') or len(row) < 3 or row[1] not in EXPECTED_IDS for row in rows):
        raise ValueError('Unknown or malformed runtime gate record')
    if any(row[0] == 'W' and row[1] != 'lixin' for row in rows):
        result.error('all', 'runtime_manifest', 'Unexpected gate walking resource')
    mesh_cache, all_meshes = {}, {}
    for gate in gates:
        ident = gate['id']
        if gate['name'] != configs[ident]['name'] or distance(gate['center_xz_m'], configs[ident]['center_xz_m']) > .001:
            result.error(ident, 'configuration', 'Manifest name or gate center differs from source configuration')
        if not all(math.isfinite(v) for key in ('center_xz_m', 'view_eye', 'view_target') for v in gate[key]):
            result.error(ident, 'configuration', 'Nonfinite camera or center')
        g_rows = [r for r in rows if r[0] == 'G' and r[1] == ident]
        if len(g_rows) != 1 or g_rows[0][2] != gate['name']:
            result.error(ident, 'runtime_manifest', 'Missing/duplicate runtime gate record')
        elif any(abs(a-float(b)) > .001 for a, b in zip(gate['view_eye']+gate['view_target']+gate['center_xz_m'], g_rows[0][3:])):
            result.error(ident, 'runtime_manifest', 'TSV camera/center differs from JSON')
        expected_layers = []
        for layer in gate['layers']:
            filename = layer['mesh']
            if Path(filename).name != filename or not filename.isascii():
                raise ValueError('Unsafe/non-ASCII mesh resource '+filename)
            if filename in mesh_cache:
                result.error(ident, 'resource', 'Shared or duplicate independent mesh '+filename)
            mesh = Mesh(assets/filename)
            mesh_cache[filename] = mesh
            if len(mesh.vertices) != layer.get('vertices', len(mesh.vertices)):
                result.error(ident, 'resource', 'Wrong vertex count '+filename)
            if not all(math.isfinite(c) and 0 <= c <= 1 for c in layer['color']):
                result.error(ident, 'resource', 'Invalid material color '+filename)
            expected_layers.append((layer['material'], filename))
        runtime_layers = [r for r in rows if r[0] == 'L' and r[1] == ident]
        if [(r[2], r[3]) for r in runtime_layers] != expected_layers:
            result.error(ident, 'runtime_manifest', 'TSV layers differ from JSON')
        for row, layer in zip(runtime_layers, gate['layers']):
            if len(row) != 8 or any(abs(float(c)-d) > 1e-5 for c, d in zip(row[5:], layer['color'])):
                result.error(ident, 'runtime_manifest', 'TSV layer color mismatch')
            expected_texture = gate.get('sign_texture') if layer['material'] == 'sign' else '-'
            if row[4] != (expected_texture or '-'):
                result.error(ident, 'runtime_manifest', 'Wrong runtime texture '+row[3])
        if gate.get('sign_texture'):
            validate_ppm(assets/gate['sign_texture'])
        if gate.get('walking_mesh'):
            mesh_cache[gate['walking_mesh']] = Mesh(assets/gate['walking_mesh'])
            if [r for r in rows if r[0] == 'W' and r[1] == ident] != [['W', ident, gate['walking_mesh']]]:
                result.error(ident, 'runtime_manifest', 'Walkable surface is not registered in TSV')
        models = [mesh_cache[layer['mesh']] for layer in gate['layers']]
        all_meshes[ident] = models
        bounds = [[min(m.bounds[0][i] for m in models) for i in range(3)],
                  [max(m.bounds[1][i] for m in models) for i in range(3)]]
        if any(abs(bounds[j][i]-gate['bounds_xyz_m'][j][i]) > .002 for j in range(2) for i in range(3)):
            result.error(ident, 'resource', 'Stored world bounds differ from actual mesh')
        if abs(bounds[1][1]-gate['top_y_m']) > .002:
            result.error(ident, 'resource', 'Stored gate top differs from actual mesh')
        if distance(gate['center_xz_m'], ((bounds[0][0]+bounds[1][0])/2, (bounds[0][2]+bounds[1][2])/2)) > 60:
            result.error(ident, 'placement', 'Actual model is remote from its configured gate')
        result.metrics[ident] = {'layers': len(models), 'vertices': sum(len(m.vertices) for m in models), 'bounds_xyz_m': bounds}
    return gates, configs, all_meshes, mesh_cache


def validate_positions(gates, configs, osm, result):
    by_id = {g['id']: g for g in gates}
    exact = {'liye': '2398019533', 'ligong': '2400176747',
             'zhixin': '7243684786', 'zhiyuan': '11762600514'}
    for ident, node in exact.items():
        error = distance(by_id[ident]['center_xz_m'], osm.nodes[node])
        result.metrics[ident]['anchor_error_m'] = error
        if error > .03:
            result.error(ident, 'mapped_anchor', f'Gate center is {error:.3f} m from entrance node/{node}')
    a, b = (osm.nodes[n] for n in ('11101744834', '11762600515'))
    if distance(by_id['zhicheng']['center_xz_m'], ((a[0]+b[0])/2, (a[1]+b[1])/2)) > .03:
        result.error('zhicheng', 'mapped_anchor', 'Gate is not between the two named east-gate barrier nodes')
    for ident, roads in ROAD_IDS.items():
        distances = {road: closest_along(osm.line(road), by_id[ident]['center_xz_m'])[0] for road in roads}
        result.metrics[ident]['mapped_road_distance_m'] = distances
        tolerance = 10 if ident in ('lide', 'zhihe', 'zhicheng') else 1
        if any(v > tolerance for v in distances.values()):
            result.error(ident, 'mapped_anchor', 'Configured gate misses mapped approach roads: '+str(distances))
    p = by_id['zhiheng']['center_xz_m']
    if closest_along(osm.line('1037951940'), p)[0] > .03:
        result.error('zhiheng', 'mapped_anchor', 'South entrance no longer lies on mapped campus boundary')
    d = closest_along(osm.line('239872139'), by_id['lixin']['center_xz_m'])[0]
    result.metrics['lixin']['mapped_steps_distance_m'] = d
    if d > 1:
        result.error('lixin', 'mapped_anchor', 'Pedestrian gate misses mapped boundary steps')
    if by_id['lizhi'].get('model_status') != 'location_marker':
        result.error('lizhi', 'evidence_boundary', 'Unconfirmed entrance is no longer labelled as a location marker')


def validate_roads(gates, osm, terrain, models, result):
    """每条已知接入车道沿中线及左右 1.25m，检查四个离地高度。

    立言门照片确认有中央岛，故检查中央岛两侧各 2.7m 的车道，而不是要求
    岛本身可以穿过。测试宽度 2.5m、高度 3.2m，不是公交或法规净空认证。
    """
    for gate in gates:
        ident = gate['id']
        checks, obstructions = 0, []
        for road in ROAD_IDS.get(ident, []):
            line = osm.line(road)
            _, nearest = closest_along(line, gate['center_xz_m'])
            length = sum(distance(a, b) for a, b in zip(line, line[1:]))
            start, stop = max(0, nearest-15), min(length, nearest+15)
            lane_centers = (-2.7, 2.7) if ident == 'liyan' else (0,)
            for lane in lane_centers:
                for lateral in (-1.25, 0, 1.25):
                    s = start
                    while s < stop-.001:
                        end = min(stop, s+1)
                        endpoints = []
                        for along in (s, end):
                            p, tangent = line_sample(line, along)
                            offset = lane+lateral
                            endpoints.append((p[0]-tangent[1]*offset, p[1]+tangent[0]*offset))
                        for height in (.45, 1.3, 2.0, 3.2):
                            a, b = [(x, terrain(x, z)+height, z) for x, z in endpoints]
                            checks += 1
                            hit = next(((m.name, p) for m in models[ident] if (p := m.first_hit(a, b)) is not None), None)
                            if hit is not None:
                                obstructions.append({'road': road, 'lane_offset_m': lane,
                                                     'lateral_sample_m': lateral, 'height_m': height,
                                                     'mesh': hit[0], 'hit_xyz_m': hit[1]})
                        s = end
        result.metrics[ident]['vehicle_rays'] = checks
        result.metrics[ident]['vehicle_hits'] = len(obstructions)
        if obstructions:
            result.error(ident, 'vehicle_clearance', {'hit_count': len(obstructions), 'first_hits': obstructions[:8]})


def validate_buildings(gates, data, osm, terrain, models, result):
    """将实存建筑轮廓内的门体三角面采样，与该建筑估计竖向范围交叉。

    检查全部层而不只岗亭的中心，故可发现屋檐/门柱进入相邻楼体。每面取
    顶点、边中点、重心；很窄的相交仍需要画面检查，不作精确布尔运算。
    """
    with (data/'building-index.csv').open(encoding='utf-8-sig', newline='') as f:
        buildings = list(csv.DictReader(f))
    for gate in gates:
        ident = gate['id']
        collisions, candidates = [], 0
        box = result.metrics[ident]['bounds_xyz_m']
        for building in buildings:
            if building['osm'] in gate.get('exclude_osm_buildings', []):
                continue
            rings = osm.rings(building['osm'])
            if not rings:
                continue
            points = [p for ring in rings for p in ring]
            bounds = (min(p[0] for p in points), max(p[0] for p in points),
                      min(p[1] for p in points), max(p[1] for p in points))
            if bounds[0] > box[1][0] or bounds[1] < box[0][0] or bounds[2] > box[1][2] or bounds[3] < box[0][2]:
                continue
            candidates += 1
            base = terrain(float(building['x_m']), float(building['z_m']))
            top = base+float(building['height_m'])
            found = None
            for mesh in models[ident]:
                for tri, tri_bounds in mesh.triangles:
                    if tri_bounds[1][1] <= base+.35 or tri_bounds[0][1] >= top-.05:
                        continue
                    samples = [v[:3] for v in tri]
                    samples += [tuple((a[k]+b[k])/2 for k in range(3))
                                for a, b in zip(tri, tri[1:]+tri[:1])]
                    samples.append(tuple(sum(v[k] for v in tri)/3 for k in range(3)))
                    for p in samples:
                        if base+.35 < p[1] < top-.05 and any(point_in_ring((p[0], p[2]), r) for r in rings):
                            found = {'osm': building['osm'], 'name': building['name'],
                                     'mesh': mesh.name, 'point_xyz_m': p, 'building_base_top_m': [base, top]}
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                collisions.append(found)
        result.metrics[ident]['nearby_building_candidates'] = candidates
        if collisions:
            result.error(ident, 'building_overlap', collisions)


def validate_lixin(gates, configs, meshes, models, terrain, result):
    gate = next(g for g in gates if g['id'] == 'lixin')
    config = configs['lixin']
    walk = meshes[gate['walking_mesh']].vertices
    cx, cz = config['center_xz_m']
    ux, uz = config['across_xz']
    vx, vz = config['outward_xz']
    def xz(x, z):
        return cx+x*ux+z*vx, cz+x*uz+z*vz
    def height(x, z):
        hits = surface_hits(walk, *xz(x, z))
        return max(hits) if hits else None
    samples, buried, gaps, blocked = [], [], [], []
    # 中央净空检查延伸至新增两端过渡的真正边界，保留全部旧检查范围。
    for lateral in (-.65, 0, .65):
        previous = None
        for j in range(261):
            local_z = 3.58-(3.58+22.38)*j/260
            x, z = xz(lateral, local_z)
            y = height(lateral, local_z)
            if y is None:
                gaps.append([lateral, local_z])
                previous = None
                continue
            clearance = y-terrain(x, z)
            samples.append(clearance)
            if clearance < -.025:
                buried.append({'local_xz_m': [lateral, local_z], 'height_minus_ground_m': clearance})
            if previous is not None:
                for body_height in (.45, 1.3, 2.0):
                    a = (previous[0], previous[1]+body_height, previous[2])
                    b = (x, y+body_height, z)
                    hit = next(((m.name, p) for m in models['lixin'] if (p := m.first_hit(a, b)) is not None), None)
                    if hit:
                        blocked.append({'local_xz_m': [lateral, local_z], 'height_m': body_height,
                                        'mesh': hit[0], 'hit_xyz_m': hit[1]})
                if abs(y-previous[1]) > .23:
                    gaps.append({'local_xz_m': [lateral, local_z], 'step_height_m': y-previous[1]})
            previous = (x, y, z)
    wide_offsets = (-4.69, -4.45, -3.5, -2.35, -1.4, -.65, 0, .65,
                    1.4, 2.35, 3.5, 4.45, 4.69)
    narrow_offsets = (-1.55, -1.4, -.65, 0, .65, 1.4, 1.55)
    connections = []
    # 新外边界的内侧必须存在 walking，外侧读邻地；全宽最多容许 6cm 接缝，
    # 不用原先 25cm 单步阈值掩盖过渡坡板仍悬空的边缘。
    for label, boundary_z, direction, offsets in (
            ('outside_entry', 3.6, 1, wide_offsets),
            ('campus_end', -22.4, -1, narrow_offsets)):
        inside_z, outside_z = boundary_z-direction*.01, boundary_z+direction*.01
        for lateral in offsets:
            y = height(lateral, inside_z)
            if y is None:
                result.error('lixin', 'walk_connection', f'Missing {label} walking surface at local x={lateral}')
                continue
            outside_walk = height(lateral, outside_z)
            ground = terrain(*xz(lateral, outside_z))
            outside_y = ground if outside_walk is None else outside_walk
            item = {'end': label, 'boundary_local_z_m': boundary_z, 'local_x_m': lateral,
                    'inside_world_xz_m': list(xz(lateral, inside_z)), 'walk_height_m': y,
                    'outside_surface': 'terrain' if outside_walk is None else 'walking',
                    'outside_surface_height_m': outside_y, 'outside_ground_m': ground,
                    'step_m': y-outside_y}
            connections.append(item)
            if abs(item['step_m']) > .06:
                result.error('lixin', 'walk_connection', item)
    result.metrics['lixin']['walk_connections'] = connections

    # 旧地面连接位置现为内接缝，必须独立比较其两侧 walking，而不是把采样点
    # 移向新边界后漏掉旧问题。保留原始离地高差便于对照，但不将桥板离地本身
    # 当成断路；断路由两侧步行面的实际高差与缺失决定。
    seams = []
    for label, seam_z, offsets in (('outside_original_step', 1.46, wide_offsets),
                                  ('campus_original_end', -20.0, narrow_offsets)):
        for lateral in offsets:
            a, b = height(lateral, seam_z-.01), height(lateral, seam_z+.01)
            item = {'seam': label, 'local_x_m': lateral, 'local_z_m': seam_z,
                    'minus_side_walking_height_m': a, 'plus_side_walking_height_m': b}
            if a is None or b is None:
                result.error('lixin', 'walk_internal_seam', item)
            else:
                item['step_m'] = b-a
                item['minus_side_height_minus_ground_m'] = a-terrain(*xz(lateral, seam_z-.01))
                item['plus_side_height_minus_ground_m'] = b-terrain(*xz(lateral, seam_z+.01))
                if abs(b-a) > .06:
                    result.error('lixin', 'walk_internal_seam', item)
            seams.append(item)
    result.metrics['lixin']['walk_internal_seams'] = seams

    # 全宽穿地检查覆盖新坡板、原来宽台阶及门厅、原内阶、藤棚和末端过渡。
    # 不以中央的可通行带代替全宽；边角、原来 +4.45m 等不平位置继续取样。
    full_width_clearances, full_width_buried, full_width_missing = [], [], []
    for label, za, zb, offsets in (
            ('outside_transition', 3.59, 1.47, wide_offsets),
            ('original_wide_steps', 1.45, -.19, wide_offsets),
            ('original_landing', -.21, -3.99, (-5.09, -4.69, -4.45, -3.5, -2.35,
                                                -1.4, -.65, 0, .65, 1.4, 2.35, 3.5, 4.45, 4.69, 5.09)),
            ('original_internal_steps', -4.01, -12.79, (-1.39, -.65, 0, .65, 1.39)),
            ('original_pergola', -12.81, -19.99, narrow_offsets),
            ('campus_transition', -20.01, -22.39, narrow_offsets)):
        count = max(1, math.ceil(abs(zb-za)/.15))
        for lateral in offsets:
            for j in range(count+1):
                local_z = za+(zb-za)*j/count
                y = height(lateral, local_z)
                if y is None:
                    full_width_missing.append({'part': label, 'local_xz_m': [lateral, local_z]})
                    continue
                clearance = y-terrain(*xz(lateral, local_z))
                full_width_clearances.append(clearance)
                if clearance < -.025:
                    full_width_buried.append({'part': label, 'local_xz_m': [lateral, local_z],
                                             'height_minus_ground_m': clearance})
    result.metrics['lixin']['walk_full_width_samples'] = len(full_width_clearances)
    result.metrics['lixin']['walk_full_width_ground_clearance_range_m'] = (
        [min(full_width_clearances), max(full_width_clearances)] if full_width_clearances else [])
    if full_width_missing:
        result.error('lixin', 'walk_full_width_missing', {'count': len(full_width_missing),
                                                        'first_samples': full_width_missing[:12]})
    if full_width_buried:
        result.error('lixin', 'walk_full_width_buried', {'count': len(full_width_buried),
                                                       'first_samples': full_width_buried[:12]})
    result.metrics['lixin']['walk_ground_clearance_range_m'] = [min(samples), max(samples)] if samples else []
    result.metrics['lixin']['walk_samples'] = len(samples)
    if buried:
        result.error('lixin', 'walk_buried', {'count': len(buried), 'first_samples': buried[:10]})
    if gaps:
        result.error('lixin', 'walk_gap_or_step', {'count': len(gaps), 'first_samples': gaps[:10]})
    if blocked:
        result.error('lixin', 'walk_obstruction', {'count': len(blocked), 'first_hits': blocked[:10]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=ROOT/'assets')
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = Check()
    try:
        gates, configs, models, meshes = validate_manifests(args.assets, args.data, result)
        terrain = Terrain(args.assets)
        osm = OSM(args.data/'yuehai.osm')
        validate_positions(gates, configs, osm, result)
        validate_roads(gates, osm, terrain, models, result)
        validate_buildings(gates, args.data, osm, terrain, models, result)
        validate_lixin(gates, configs, meshes, models, terrain, result)
    except (ValueError, KeyError, OSError, struct.error) as exc:
        result.error('all', 'input_or_format', str(exc))
    report = {'passed': not result.errors, 'errors': result.errors,
              'warnings': result.warnings, 'gates': result.metrics,
              'limits': ['Vehicle corridors sample 2.5 m width and heights 0.45/1.3/2.0/3.2 m; they are not measured road clearances.',
                         'OSM building volume tests use estimated base heights and triangle samples, not a mesh boolean.',
                         'Configuration agreement does not turn approximate entrances into surveyed positions.',
                         'No visual/runtime verification or real-world opening-state inference is performed.']}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for ident, item in result.metrics.items():
        print(f'{ident}: {item["layers"]} layers, {item["vertices"]} vertices, '
              f'{item.get("vehicle_rays", 0)} vehicle rays, {item.get("vehicle_hits", 0)} hits')
    for item in result.errors:
        print('ERROR '+item['gate']+' / '+item['check']+': '+json.dumps(item['detail'], ensure_ascii=False))
    for item in result.warnings:
        print('WARNING '+item['gate']+' / '+item['check']+': '+json.dumps(item['detail'], ensure_ascii=False))
    print(f'Gate asset checks: {len(result.errors)} errors, {len(result.warnings)} warnings')
    return 1 if result.errors else 0


if __name__ == '__main__':
    sys.exit(main())
