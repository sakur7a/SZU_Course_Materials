#!/usr/bin/env python3
"""宿舍最终资产的独立检查；仅用标准库及项目内的标准库几何读取器。

不导入生成器，不启动 OpenGL。读取 campus-{qiaoyuan,south}-dorms.json/tsv、
SZM1 网格、OSM、地点绑定及地形；检查各楼独立资源、单位法线与非退化面绕序，
并用少量几何射线核对乔苑凹阳台、屋顶斜板支架和乔相阁园路入口。
terrain_draped入口还需可见slab与walk的实际三角面一致、内部贴地及地形上方
1.3/2.0米成人净空；未声明新模式的旧flat资源仍按旧平面规则核验。

本检查核对程序资产及建模约束，不证明照片细节、实际门位、现实开放状态或
楼高估计已经实测。阳台/支架为抽样，不代替全面碰撞、整条道路及 GUI 验证。
"""
import argparse
import ast
import collections
import csv
import json
import math
import sys
from pathlib import Path

from verify_gate_assets import Mesh, Terrain, cross, dot, segment_triangle, sub, surface_hits, validate_ppm
from verify_place_assets import OsmSnapshot, distance_to_segment, ring_location


ROOT = Path(__file__).resolve().parent
GROUP_IDS = {
    'qiaoyuan': {'qiaomu', 'qiaolin', 'qiaosen', 'qiaoxiang', 'qiaowu'},
    'south': {'chundi', 'xiazheng', 'gaoyin', 'qiuse', 'dongzhu', 'diyin'},
}
# 原OSM连桥端点；不能从新portal自己的metadata反推要检查哪些门。
SOUTH_SKYWALK_ENDPOINTS = {'xiazheng': ('776358619', 'end'),
                           'dongzhu': ('776358645', 'end'),
                           'qiuse': ('776358645', 'start')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(values, count, label):
    require(isinstance(values, (list, tuple)) and len(values) == count
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values),
            f'{label}: expected {count} finite numbers')
    return values


def unique(items, label):
    result = {}
    for item in items:
        require(item['id'] not in result, f'{label}: duplicate id {item["id"]}')
        result[item['id']] = item
    return result


def resource_path(assets, filename):
    require(isinstance(filename, str) and filename and filename.isascii()
            and Path(filename).name == filename and '/' not in filename and '\\' not in filename,
            f'Unsafe or non-ASCII resource name: {filename!r}')
    path = assets / filename
    require(path.is_file(), f'Missing final asset: {path}')
    return path


def layer_row(layer):
    return ('L', layer['id'], layer['material'], layer['mesh'], layer.get('texture') or '-',
            *(round(float(value), 6) for value in finite(layer['color'], 3, layer['mesh'] + '/color')))


def validate_manifest(assets, data, group, owners, checked_textures):
    manifest_path = assets / f'campus-{group}-dorms.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    config = json.loads((data / f'{group}-dorms.json').read_text(encoding='utf-8'))
    buildings = unique(manifest['buildings'], group + ' manifest')
    configs = unique(config['buildings'], group + ' source')
    require(buildings.keys() == configs.keys() == GROUP_IDS[group], f'{group}: missing or unexpected dormitory ids')
    require(len({item['osm'] for item in buildings.values()}) == len(buildings), f'{group}: duplicated OSM building')
    own_layers = [layer for item in buildings.values() for layer in item['layers']]
    require(collections.Counter(map(layer_row, own_layers)) == collections.Counter(map(layer_row, manifest['layers'])),
            f'{group}: flat and per-building layers disagree')
    expected_walks = [('W', item['id'], item['walking_mesh']) for item in buildings.values() if item.get('walking_mesh')]
    elevated, expected_views = [], []
    for ident, building in buildings.items():
        skywalks = building.get('skywalks', [])
        expected_ids = {ident + '-skywalk'} if group == 'south' and ident in SOUTH_SKYWALK_ENDPOINTS else set()
        require({item['id'] for item in skywalks} == expected_ids and len(skywalks) == len(expected_ids),
                f'{ident}: missing or unexpected mapped elevated entrance')
        require(building.get('elevated_walking_meshes', []) == [item['walking_mesh'] for item in skywalks],
                f'{ident}: elevated walking files differ from its mapped portals')
        for skywalk in skywalks:
            require(skywalk['owner_id'] == ident, f'{ident}: elevated entrance belongs to another building')
            expected_walks.append(('W', skywalk['id'], skywalk['walking_mesh']))
            elevated.append(skywalk)
            if 'view_eye' in skywalk or 'view_target' in skywalk:
                eye = finite(skywalk.get('view_eye'), 3, skywalk['id'] + '/view_eye')
                target = finite(skywalk.get('view_target'), 3, skywalk['id'] + '/view_target')
                view_id = skywalk.get('walking_id', skywalk['id'])
                require(view_id == ident + '-skywalk', f'{ident}: unknown elevated view id')
                expected_views.append(('G', view_id, skywalk.get('view_name', building['name'] + '高架入口'),
                                       *(round(value, 6) for value in [*eye, *target, target[0], target[2]])))
    if expected_views:
        require(group == 'south' and {row[1] for row in expected_views}
                == {ident + '-skywalk' for ident in SOUTH_SKYWALK_ENDPOINTS},
                f'{group}: the three elevated entrance views must be provided together')
    require(collections.Counter(expected_walks) == collections.Counter(('W', w['id'], w['mesh']) for w in manifest['walking_meshes']),
            f'{group}: flat and per-building walking files disagree')
    flat_walks = unique(manifest['walking_meshes'], group + ' walking registry')
    for skywalk in elevated:
        registry = flat_walks[skywalk['id']]
        require(registry.get('owner_id') == skywalk['owner_id'] and registry.get('source_osm') == skywalk['source_osm'],
                f'{skywalk["id"]}: elevated walking ownership/source differs from the portal')
    rows = []
    with (assets / f'campus-{group}-dorms.tsv').open(encoding='utf-8', newline='') as stream:
        for row in csv.reader(stream, delimiter='\t'):
            if not row or row[0].startswith('#'):
                continue
            sizes = {'L': 8, 'W': 3, 'G': 11}
            require(row[0] in sizes and len(row) == sizes[row[0]], f'{group}: malformed runtime row {row!r}')
            if row[0] == 'L':
                colors = tuple(float(value) for value in row[5:])
                require(all(math.isfinite(value) and 0 <= value <= 1 for value in colors), f'{group}: invalid runtime color')
                rows.append((*row[:5], *(round(value, 6) for value in colors)))
            elif row[0] == 'G':
                require(group == 'south' and row[1] in {ident + '-skywalk' for ident in SOUTH_SKYWALK_ENDPOINTS},
                        f'{group}: unknown elevated camera row {row!r}')
                numbers = tuple(float(value) for value in row[3:])
                require(all(math.isfinite(value) for value in numbers), f'{group}: nonfinite elevated camera row')
                rows.append((*row[:3], *(round(value, 6) for value in numbers)))
            else:
                rows.append(tuple(row))
    require(collections.Counter(rows) == collections.Counter([*map(layer_row, own_layers), *expected_walks, *expected_views]),
            f'{group}: runtime L/W/G rows differ from the real building resource list')
    for ident, building in buildings.items():
        prefix = f'campus-qiaoyuan-{ident}-' if group == 'qiaoyuan' else f'campus-dorm-{ident}-'
        roles = [layer['role'] for layer in building['layers']]
        require(len(set(roles)) == len(roles) and {'wall', 'glass', 'metal', 'roof'} <= set(roles),
                f'{ident}: missing or duplicated structural material roles')
        files = [*building['layers'], *([{'mesh': building['walking_mesh']}] if building.get('walking_mesh') else []),
                 *[{'mesh': filename} for filename in building.get('elevated_walking_meshes', [])]]
        for item in files:
            filename = item['mesh']
            require(filename.startswith(prefix) and filename.endswith('.mesh'), f'{ident}: mixed independent building file {filename}')
            require(filename not in owners, f'{ident}: mesh is shared or repeated: {filename} (owner {owners.get(filename)})')
            owners[filename] = ident
            resource_path(assets, filename)
        for layer in building['layers']:
            require(layer['id'] == ident, f'{ident}: layer belongs to {layer["id"]}')
            color = finite(layer['color'], 3, layer['mesh'] + '/color')
            require(all(0 <= value <= 1 for value in color), f'{ident}: invalid layer color')
            texture = layer.get('texture')
            if texture and texture not in checked_textures:
                validate_ppm(resource_path(assets, texture))
                checked_textures.add(texture)
    return manifest, buildings, configs, config


def mesh_quality(mesh):
    """逐个非退化面比较几何法线与存储法线；不靠取消背面裁剪掩盖反绕序。"""
    degenerate = 0
    for tri, _ in mesh.triangles:
        normal = cross(sub(tri[1][:3], tri[0][:3]), sub(tri[2][:3], tri[0][:3]))
        length = math.sqrt(dot(normal, normal))
        if length < 0.0000001:
            degenerate += 1
            continue
        require(all(dot(normal, vertex[3:6]) / length > .97 for vertex in tri),
                f'{mesh.name}: triangle winding does not match stored normals')
    require(degenerate < len(mesh.triangles), f'{mesh.name}: no nondegenerate geometry')
    return degenerate


def footprint_distance(point, area):
    outer, inner = area
    if any(ring_location(point, ring) >= 0 for ring in outer) and not any(ring_location(point, ring) > 0 for ring in inner):
        return 0.0
    return min(distance_to_segment(point, a, b) for ring in outer + inner for a, b in zip(ring, ring[1:]))


def school_floor_override(building, source, data):
    """学校一期12层记录优先于 OSM 误记；39.6m仍只作为建模估计核对。

    只读取 HEIGHTS 的 AST 字面量，绝不导入或执行生成器。
    """
    ident = building['id']
    require(source['phase'] == 1 and source['floors'] == building['floors'] == 12
            and abs(source['nominal_height_m'] - 39.6) < .001
            and abs(building['height_m'] - 39.6) < .001,
            f'{ident}: school floor override requires 12 floors and the documented 39.6m estimate')
    document = (data / 'qiaoyuan-dorms-sources.md').read_text(encoding='utf-8')
    require('https://www.szu.edu.cn/info/1161/10976.htm' in document
            and '主体12层' in document and '乔木' in document and '乔林' in document,
            f'{ident}: school floor override has lost its explicit first-phase source')
    tree = ast.parse((data.parent / 'generate_campus.py').read_text(encoding='utf-8'))
    heights = next((node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'HEIGHTS' for target in node.targets)), None)
    require(isinstance(heights, ast.Dict), f'{ident}: cannot independently read HEIGHTS literal')
    entry = next((ast.literal_eval(value) for key, value in zip(heights.keys, heights.values)
                  if isinstance(key, ast.Constant) and key.value == building['osm']), None)
    require(entry is not None and abs(entry[0] - 39.6) < .001 and '12层' in entry[1],
            f'{ident}: HEIGHTS no longer matches the school 12-floor / estimated 39.6m override')


def check_binding(building, source, osm, places, data):
    ident = building['id']
    require((building['osm'], building['name'], building['floors']) == (source['osm'], source['name'], source['floors']),
            f'{ident}: id/OSM/name/floor metadata disagree with the source catalogue')
    element = osm.elements[building['osm']]
    tags = {tag.get('k'): tag.get('v') for tag in element.findall('tag')}
    if ident in places:
        require((places[ident].get('osm'), places[ident]['name']) == (building['osm'], building['name']),
                f'{ident}: authoritative campus-places binding changed')
        binding = 'campus-places'
    else:
        # 高音/低音可能还没有导航标签；建筑自身必须与其 OSM 名称相符。
        require(tags.get('name') == building['name'], f'{ident}: no place binding and OSM name disagrees')
        binding = 'OSM (no navigation label)'
    if ident in ('qiaomu', 'qiaolin'):
        school_floor_override(building, source, data)
    elif tags.get('building:levels'):
        require(int(tags['building:levels']) == building['floors'], f'{ident}: OSM floor count disagrees')
    return binding


def triangle_key(triangle):
    # 两个文件保存同一批float32顶点；忽略绕序，以免把正常的面方向修正当作缺面。
    return tuple(sorted(tuple(vertex[:3]) for vertex in triangle))


def qiaoyuan_walking_mode(building):
    mode = building.get('walking_surface', 'flat')
    require(mode in ('flat', 'terrain_draped'), f'{building["id"]}: unsupported walking mode {mode!r}')
    return mode


def entry_band_contains(point, building):
    """入口路径的窄带，包括折线外侧的正常斜接角，不扩大整栋墙体的范围。"""
    half = building['walking_path_half_width_m']
    require(math.isfinite(half) and abs(half - 1.05) < .001,
            f'{building["id"]}: unexpected entrance paving half-width')
    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]
    for line in building['entry_paths_xz_m']:
        if any(distance_to_segment(point, a, b) <= half + .003 for a, b in zip(line, line[1:])):
            return True
        # buffer(join_style=mitre)的折点尖角可能比half宽；只接受实际偏移边的交点三角区。
        for a, corner, b in zip(line, line[1:], line[2:]):
            first, second = sub(corner, a), sub(b, corner)
            l0, l1 = math.hypot(*first), math.hypot(*second)
            if min(l0, l1) < .000001:
                continue
            first, second = tuple(x / l0 for x in first), tuple(x / l1 for x in second)
            denominator = cross2(first, second)
            if abs(denominator) < .000001:
                continue
            for side in (-1, 1):
                p0 = (corner[0] - first[1] * half * side, corner[1] + first[0] * half * side)
                p1 = (corner[0] - second[1] * half * side, corner[1] + second[0] * half * side)
                along = cross2(sub(p1, p0), second) / denominator
                joint = (p0[0] + along * first[0], p0[1] + along * first[1])
                if math.dist(joint, corner) <= half * 5 + .003 and ring_location(point, [p0, joint, p1, p0]) >= 0:
                    return True
    return False


def check_south_skywalk(building, catalog, registry, walking, models, osm, bridge):
    """从原OSM端点、方向和正式桥面独立重建三门审计，不采用portal自报高度。"""
    ident = building['id']
    way, endpoint = SOUTH_SKYWALK_ENDPOINTS[ident]
    expected_source = 'way/' + way
    sources = unique(catalog['skywalk_interfaces'], 'source skywalk interfaces')
    require(set(sources) == {owner + '-skywalk' for owner in SOUTH_SKYWALK_ENDPOINTS},
            'Source catalogue must retain all three mapped elevated entrances')
    interface = building['skywalks'][0]
    source = sources[ident + '-skywalk']
    require(interface['owner_id'] == source['owner_id'] == ident
            and interface['source_osm'] == source['source_osm'] == expected_source
            and interface['endpoint'] == source['endpoint'] == endpoint,
            f'{ident}: elevated source or endpoint differs from the independent OSM binding')
    for field in ('clear_width_m', 'clear_height_m', 'inside_length_m', 'outside_overlap_m', 'platform_thickness_m'):
        finite([interface[field], source[field]], 2, ident + '/' + field)
        require(abs(interface[field] - source[field]) < .001, f'{ident}: elevated configuration mismatch for {field}')
    require(2.5 <= source['clear_width_m'] <= 4 and source['clear_height_m'] >= 2
            and 3 <= source['inside_length_m'] <= 4.5 and .1 <= source['outside_overlap_m'] <= 1,
            f'{ident}: elevated dimensions do not retain the audited passage')
    line = [osm.nodes[node] for node in osm.ways[way]]
    edge, neighbor = (line[-1], line[-2]) if endpoint == 'end' else (line[0], line[1])
    direction = sub(edge, neighbor)
    length = math.hypot(*direction)
    inward = tuple(value / length for value in direction)
    lateral = (-inward[1], inward[0])
    require(math.dist(edge, finite(interface['edge_xz_m'], 2, ident + '/edge')) < .015
            and math.dist(inward, finite(interface['inward_xz'], 2, ident + '/inward')) < .001,
            f'{ident}: portal does not align with the original OSM endpoint/tangent')
    def xz(s, across=0):
        return (edge[0] + inward[0] * s + lateral[0] * across,
                edge[1] + inward[1] * s + lateral[1] * across)
    def local(point):
        delta = sub(point, edge)
        return dot(delta, inward), dot(delta, lateral)
    require(osm.covers(building['osm'], xz(3)), f'{ident}: independently mapped inward tangent misses the building')
    bridge_cache = {}
    def bridge_y(point, optional=False):
        key = tuple(round(value, 6) for value in point)
        if key not in bridge_cache:
            hits = surface_hits(bridge.vertices, *point)
            require(not hits or max(hits) - min(hits) < .04, f'{ident}: ambiguous original bridge deck at {point}')
            bridge_cache[key] = sum(hits) / len(hits) if hits else None
        height = bridge_cache[key]
        require(height is not None or optional, f'{ident}: original bridge mesh has no deck at {point}')
        return height
    deck = bridge_y(edge)
    require(abs(interface['deck_y_m'] - deck) < .015 and abs(registry['deck_y_m'] - deck) < .015,
            f'{ident}: elevated metadata is not the actual bridge endpoint height')
    require(abs(interface['platform_top_y_m'] - (deck + .012)) < .015,
            f'{ident}: internal platform is not aligned with the actual bridge endpoint')
    width = source['clear_width_m'] + .30
    require(abs(interface['platform_width_m'] - width) < .001, f'{ident}: platform width does not preserve railing clearance')
    outside, inside = source['outside_overlap_m'], source['inside_length_m']
    declared_path = interface['path_xz_m']
    require(len(declared_path) == 3 and all(math.dist(finite(point, 2, ident + '/skywalk path'), expected) < .015
                                          for point, expected in zip(declared_path, (xz(-outside), edge, xz(inside)))),
            f'{ident}: elevated path does not retain the independently mapped bridge/inside endpoints')
    polygon = [tuple(finite(point, 2, ident + '/platform polygon')) for point in interface['platform_polygon_xz_m']]
    require(len(polygon) >= 3, f'{ident}: no elevated platform polygon')
    if polygon[-1] != polygon[0]:
        polygon.append(polygon[0])
    def canonical_plan(point, margin=.004):
        s, across = local(point)
        return -outside - margin <= s <= inside + margin and abs(across) <= width / 2 + margin
    def platform_plan(point, margin=.004):
        return (ring_location(point, polygon) >= 0
                or min(distance_to_segment(point, a, b) for a, b in zip(polygon, polygon[1:])) <= margin)
    require(all(canonical_plan(point) for point in polygon), f'{ident}: declared platform extends beyond its mapped entrance rectangle')
    def expected_top(point):
        s, _ = local(point)
        if s >= -1e-5:
            return deck + .012
        # 外侧沿正式桥面坡度接合；桥外宽出的角点沿同中心线剖面延续。
        height = bridge_y(point, optional=True)
        if height is None:
            height = bridge_y(xz(s))
        return height + .012
    keys = {triangle_key(triangle) for triangle, _ in walking.triangles}
    visible = [triangle for triangle, _ in models['paving'].triangles if triangle_key(triangle) in keys]
    require({triangle_key(triangle) for triangle in visible} == keys,
            f'{ident}: elevated walk triangles are absent from the visible paving layer')
    visual_vertices = [vertex for triangle in visible for vertex in triangle]
    probes = set()
    for vertex in walking.vertices:
        point = (vertex[0], vertex[2])
        require(canonical_plan(point) and platform_plan(point), f'{ident}: elevated W leaves its independently mapped platform')
        require(abs(vertex[1] - expected_top(point)) < .025, f'{ident}: elevated W uses an incorrect bridge/ground height')
    for triangle, _ in walking.triangles:
        for weights in ((1/3, 1/3, 1/3), (.6, .2, .2), (.2, .6, .2), (.2, .2, .6)):
            probes.add(tuple(sum(vertex[axis] * weight for vertex, weight in zip(triangle, weights)) for axis in (0, 2)))
    for along in (-outside, -.2, -.05, 0, .1, .5, 1, 2, 2.95, 3):
        for across in (-source['clear_width_m'] / 2 + .10, 0, source['clear_width_m'] / 2 - .10):
            probes.add(xz(along, across))
    max_bridge_error = max_visual_gap = 0.0
    for point in probes:
        hits = surface_hits(walking.vertices, *point)
        require(hits, f'{ident}: elevated platform has a gap at {point}')
        expected = expected_top(point)
        error = max(abs(height - expected) for height in hits)
        require(error < .05, f'{ident}: elevated platform/bridge seam differs by {error:.3f}m at {point}')
        max_bridge_error = max(max_bridge_error, error)
        visual = surface_hits(visual_vertices, *point)
        require(visual, f'{ident}: elevated W has no visible paving at {point}')
        gap = max(min(abs(height - other) for other in visual) for height in hits)
        require(gap < .008, f'{ident}: elevated visible/walk surface mismatch')
        max_visual_gap = max(max_visual_gap, gap)
    profile = interface['approach_profile']
    require(len(profile) >= 2 and abs(profile[0]['s_m'] + outside) < .015 and abs(profile[-1]['s_m']) < .015,
            f'{ident}: elevated approach profile omits its actual bridge-side end')
    for item in profile:
        require(-outside - .001 <= item['s_m'] <= .001, f'{ident}: invalid bridge approach station')
        point = xz(item['s_m'])
        original = bridge_y(point)
        require(abs(item['bridge_y_m'] - original) < .015 and abs(item['platform_y_m'] - expected_top(point)) < .025,
                f'{ident}: approach profile is inconsistent with the independent bridge deck')
    # 复现原只读审计的9条射线：OSM端点外0.65m至内3m，deck来自原桥网格。
    audit_rays = []
    for height in (.45, 1.3, 1.8):
        a, b = xz(-.65), xz(3)
        start, end = (a[0], deck + height, a[1]), (b[0], deck + height, b[1])
        hit = first_hit(models, start, end)
        require(hit is None, f'{ident}: independent OSM bridge entrance ray at deck+{height}m is blocked: {hit}')
        audit_rays.append({'height_above_bridge_endpoint_m': height, 'start_xyz_m': start, 'end_xyz_m': end})
    # 除中心线回归外，按实际平台高度检查声明净宽内的成人头部空间。
    for across in (-source['clear_width_m'] / 2 + .15, 0, source['clear_width_m'] / 2 - .15):
        for along in (0, .5, 1, 2, 2.95):
            point = xz(along, across)
            floor = expected_top(point)
            hit = first_hit(models, (point[0], floor + .45, point[1]),
                            (point[0], floor + source['clear_height_m'], point[1]))
            require(hit is None, f'{ident}: declared elevated entrance width/headroom is blocked: {hit}')
    camera = None
    if 'view_eye' in interface:
        eye, target = interface['view_eye'], interface['view_target']
        original = bridge_y((eye[0], eye[2]))
        require(eye[1] >= original + 1.7, f'{ident}: elevated view eye is below the real bridge walking height')
        require(surface_hits(walking.vertices, target[0], target[2]), f'{ident}: elevated view does not target the actual entrance platform')
        camera = {'eye_height_above_original_bridge_m': eye[1] - original}
    context = {'interface': interface, 'canonical_plan': canonical_plan, 'platform_plan': platform_plan,
               'minimum_y_m': walking.bounds[0][1], 'maximum_y_m': walking.bounds[1][1]}
    metrics = {'source_osm': expected_source, 'osm_endpoint': endpoint, 'edge_xz_m': edge,
               'independent_bridge_deck_y_m': deck, 'walking_mesh': interface['walking_mesh'],
               'vertices': len(walking.vertices), 'surface_probe_count': len(probes),
               'max_bridge_profile_error_m': max_bridge_error, 'max_visible_walk_gap_m': max_visual_gap,
               'audit_rays': audit_rays, 'camera': camera}
    return context, metrics


def skywalk_triangle_allowed(role, triangle, contexts):
    """只允许已独立校准的平台面/侧边和低护栏越出外墙，不扩展其它楼体。"""
    for context in contexts or []:
        if role == 'paving':
            low = context['minimum_y_m'] - context['interface']['platform_thickness_m'] - .025
            high = context['maximum_y_m'] + .025
            if all(low <= vertex[1] <= high and context['platform_plan']((vertex[0], vertex[2])) for vertex in triangle):
                return True
        elif role == 'metal':
            if all(context['minimum_y_m'] - .08 <= vertex[1] <= context['maximum_y_m'] + 1.25
                   and context['canonical_plan']((vertex[0], vertex[2]), margin=.09) for vertex in triangle):
                return True
    return False


def check_extent(building, source, models, walking, osm, terrain, group, skywalk_contexts=None):
    ident = building['id']
    base, height = building['base_y_m'], building['height_m']
    finite([base, height], 2, ident + '/height')
    require(isinstance(building['floors'], int) and 2 < building['floors'] < 50 and 2.5 < height / building['floors'] < 5,
            f'{ident}: implausible model floor/height metadata')
    if 'nominal_height_m' in source:
        require(abs(height - source['nominal_height_m']) < .05, f'{ident}: nominal model height differs from source')
    area = osm.area(building['osm'])
    bounds = [[min(m.bounds[0][i] for m in models.values()) for i in range(3)],
              [max(m.bounds[1][i] for m in models.values()) for i in range(3)]]
    if 'bounds_xyz_m' in building:
        require(all(abs(bounds[j][i] - building['bounds_xyz_m'][j][i]) < .003 for j in range(2) for i in range(3)),
                f'{ident}: stored bounds disagree with final mesh geometry')
    allowed_roof = .15 if group == 'qiaoyuan' and source['phase'] == 2 else 1.30
    require(base + height - .3 <= bounds[1][1] <= base + height + allowed_roof,
            f'{ident}: actual mesh top {bounds[1][1] - base:.3f} m is inconsistent with nominal height {height:.3f} m')
    ground_samples = [terrain(*point) for ring in area[0] for point in ring]
    ground_min, ground_max = min(ground_samples), max(ground_samples)
    require(ground_min - 2 <= base <= ground_max + 2,
            f'{ident}: declared base is remote from the existing rendered site terrain')
    require(bounds[0][1] >= min(base, ground_min) - 2.0, f'{ident}: foundation extends implausibly below its site')
    osm_bounds = osm.bounds(building['osm'])
    for axis, lo, hi in ((0, osm_bounds[0], osm_bounds[2]), (2, osm_bounds[1], osm_bounds[3])):
        require(bounds[1][axis] - bounds[0][axis] >= (hi - lo) * .85,
                f'{ident}: final model omits a substantial part of the bound footprint')
    # 检查所有不同 XZ，不只取包围盒角点；毫米量化避免对每层重复杆件重算。
    draped = group == 'qiaoyuan' and qiaoyuan_walking_mode(building) == 'terrain_draped'
    paving_keys = {triangle_key(triangle) for triangle, _ in walking.triangles} if draped and walking else set()
    if draped:
        require(building.get('walking_visible_role') == 'slab' and walking is not None,
                f'{ident}: terrain-draped walk must identify its visible slab surface')
    max_distance, points, cache, paving_points, skywalk_points = 0.0, set(), {}, set(), set()
    for role, model in [*models.items(), *([('walk', walking)] if walking else [])]:
        own_points = set()
        for triangle, _ in model.triangles:
            is_paving = draped and (role == 'walk' or (role == 'slab' and triangle_key(triangle) in paving_keys))
            is_skywalk = skywalk_triangle_allowed(role, triangle, skywalk_contexts)
            target = skywalk_points if is_skywalk else paving_points if is_paving else own_points
            for vertex in triangle:
                target.add((round(vertex[0], 3), round(vertex[2], 3)))
        points.update(own_points)
        # 乔苑门檐在角点处允许长、宽两个方向同时伸出，不能只用1.15m深度
        # 当作对角线余量；主体仍按实际阳台前缘0.65m加杆径的邻域检查。
        allowance = (2.30 if role == 'roof' else .90) if group == 'qiaoyuan' else .20
        if role == 'walk':
            allowance = .03
        for point in own_points:
            if point not in cache:
                cache[point] = footprint_distance(point, area)
            distance = cache[point]
            max_distance = max(max_distance, distance)
            require(distance <= allowance, f'{ident}/{role}: mesh point {point} lies {distance:.3f} m from its own footprint')
    # 只有与walk实际三角面一致的低位slab铺面可沿完整入口路径延伸。其余slab、
    # 主体和门柱仍执行上面的原轮廓阈值，不因新模式整体放宽。
    for point in paving_points:
        require(entry_band_contains(point, building), f'{ident}: entrance paving leaves its declared narrow path band at {point}')
    if walking:
        for vertex in walking.vertices:
            expected = (terrain(vertex[0], vertex[2]) + .035 if draped else base + .035) if group == 'qiaoyuan' else terrain(vertex[0], vertex[2]) + .025
            require(abs(vertex[1] - expected) < .015, f'{ident}: walking surface disagrees with its declared floor/terrain')
    return {'bounds_xyz_m': bounds, 'actual_max_height_m': bounds[1][1] - base,
            'site_terrain_range_m': [ground_min, ground_max],
            'max_footprint_outset_m': max_distance, 'unique_xz_points': len(points),
            'footprint_scope': 'body except independently verified entrance surfaces' if draped or skywalk_contexts else 'all model and walking layers',
            'entrance_paving_xz_points': len(paving_points), 'verified_skywalk_xz_points': len(skywalk_points)}


def check_draped_walk(building, models, walking, terrain):
    ident = building['id']
    require(building.get('walking_visible_role') == 'slab', f'{ident}: draped visible paving role must be slab')
    keys = {triangle_key(triangle) for triangle, _ in walking.triangles}
    visible = [triangle for triangle, _ in models['slab'].triangles if triangle_key(triangle) in keys]
    require({triangle_key(triangle) for triangle in visible} == keys,
            f'{ident}: visible slab is missing actual walking triangles')
    visible_vertices = [vertex for triangle in visible for vertex in triangle]
    probes = {}
    def probe(point, kind):
        key = (round(point[0], 6), round(point[1], 6))
        probes.setdefault(key, set()).add(kind)
    # 顶点贴地不足以保证大三角面的内部贴地；在每个非退化面内部取四点。
    for triangle, _ in walking.triangles:
        face_normal = cross(sub(triangle[1][:3], triangle[0][:3]), sub(triangle[2][:3], triangle[0][:3]))
        if dot(face_normal, face_normal) < .00000000000001:
            continue
        for weights in ((1/3, 1/3, 1/3), (.6, .2, .2), (.2, .6, .2), (.2, .2, .6)):
            point = tuple(sum(vertex[axis] * weight for vertex, weight in zip(triangle, weights)) for axis in (0, 2))
            probe(point, 'triangle_interior')
    for line in building['entry_paths_xz_m']:
        require(len(line) >= 2, f'{ident}: missing entrance centreline')
        probe(line[0], 'path_endpoint');probe(line[-1], 'path_endpoint')
        for a, b in zip(line, line[1:]):
            finite(a, 2, 'entrance point');finite(b, 2, 'entrance point')
            pieces = max(2, math.ceil(math.dist(a, b) / .5))
            for index in range(pieces + 1):
                probe(tuple(a[axis] + (b[axis] - a[axis]) * index / pieces for axis in (0, 1)), 'path_centreline')
    require(probes, f'{ident}: no actual entrance floor probes')
    max_terrain_error = max_visible_gap = 0.0
    offset_min, offset_max = math.inf, -math.inf
    counts = collections.Counter()
    for point, kinds in probes.items():
        counts.update(kinds)
        require(entry_band_contains(point, building), f'{ident}: entrance triangle interior leaves the path band at {point}')
        ground = terrain(*point)
        expected = ground + .035
        hits = surface_hits(walking.vertices, *point)
        require(hits, f'{ident}: walking mesh does not cover entrance probe {point} ({sorted(kinds)})')
        error = max(abs(height - expected) for height in hits)
        require(error < .05, f'{ident}: walking triangle interior/path differs from terrain+.035 by {error:.3f}m at {point}')
        max_terrain_error = max(max_terrain_error, error)
        offset_min = min(offset_min, *(height - ground for height in hits))
        offset_max = max(offset_max, *(height - ground for height in hits))
        visible_hits = surface_hits(visible_vertices, *point)
        require(visible_hits, f'{ident}: no visible slab paving at walking probe {point}')
        gap = max(min(abs(height - visual) for visual in visible_hits) for height in hits)
        require(gap < .008, f'{ident}: visible slab and walk differ by {gap:.3f}m at {point}')
        max_visible_gap = max(max_visible_gap, gap)
    return {'mode': 'terrain_draped', 'visible_role': 'slab', 'probes_by_kind': dict(counts),
            'height_above_terrain_range_m': [offset_min, offset_max],
            'max_triangle_terrain_error_m': max_terrain_error, 'max_visible_walk_gap_m': max_visible_gap}


def check_grounded_entry(building, models, osm, terrain):
    """按脚下真实地形检查成人身高，不能靠抬高/降低宿舍base掩盖入口顶头。"""
    ident = building['id']
    segments, enters, head_points = [], False, set()
    for line in building['entry_paths_xz_m']:
        for a, b in zip(line, line[1:]):
            length = math.dist(a, b)
            if length < .001:
                continue
            normal = (-(b[1] - a[1]) / length, (b[0] - a[0]) / length)
            pieces = max(1, math.ceil(length / .5))
            for offset in (-.5, 0, .5):
                for index in range(pieces):
                    ends = [tuple(a[axis] + (b[axis] - a[axis]) * t + normal[axis] * offset for axis in (0, 1))
                            for t in (index / pieces, (index + 1) / pieces)]
                    enters |= any(osm.covers(building['osm'], point) for point in ends)
                    head_points.update(ends)
                    for height in (1.3, 2.0):
                        segments.append(tuple((point[0], terrain(*point) + height, point[1]) for point in ends))
    require(enters and segments, f'{ident}: entrance paths do not enter their own footprint')
    path_segments = len(segments)
    # 补竖向头部探针，捕捉恰与水平头顶射线共面的低门楣；不能把共面漏检视为净空。
    for point in head_points:
        ground = terrain(*point)
        segments.append(((point[0], ground + 1.3, point[1]), (point[0], ground + 2.0, point[1])))
    minimum = min(point[1] for segment in segments for point in segment)
    maximum = max(point[1] for segment in segments for point in segment)
    triangles = [(role, triangle, bounds) for role, model in models.items() for triangle, bounds in model.triangles
                 if bounds[0][1] <= maximum + 1e-5 and bounds[1][1] >= minimum - 1e-5]
    for start, end in segments:
        lower = [min(a, b) - 1e-5 for a, b in zip(start, end)]
        upper = [max(a, b) + 1e-5 for a, b in zip(start, end)]
        for role, triangle, bounds in triangles:
            if any(bounds[0][axis] > upper[axis] or bounds[1][axis] < lower[axis] for axis in range(3)):
                continue
            hit = segment_triangle(start, end, triangle)
            if hit is not None:
                height = hit[1] - terrain(hit[0], hit[2])
                raise ValueError(f'{ident}: terrain-relative adult entrance ray is blocked by {role} at {hit}, '
                                 f'{height:.3f}m above terrain')
    return {'segments_at_body_height': path_segments, 'vertical_head_probes': len(head_points),
            'height_above_terrain_m': [1.3, 2.0],
            'lateral_offsets_m': [-.5, 0, .5], 'max_segment_length_xz_m': .5}


def first_hit(models, start, end):
    for role, model in models.items():
        if (point := model.first_hit(start, end)) is not None:
            return {'role': role, 'point_xyz_m': point}
    return None


def check_solar(building, source, models):
    count = building['solar_panels']
    if not source.get('solar'):
        require(count == 0 and 'solar' not in models, f'{building["id"]}: unconfigured rooftop panel layer')
        return {'panels': 0, 'support_samples': []}
    require(count > 0 and 'solar' in models, f'{building["id"]}: rooftop panel geometry is missing')
    panels = models['solar'].vertices
    require(len(panels) == count * 6, f'{building["id"]}: rooftop panel count does not match actual quads')
    for index in range(count):
        vertices = panels[index * 6:(index + 1) * 6]
        require(max(v[1] for v in vertices) - min(v[1] for v in vertices) > .20
                and all(.1 < vertex[4] < .99 for vertex in vertices),
                f'{building["id"]}: rooftop panel {index} is flat or incorrectly oriented')
    roof = building['base_y_m'] + building['height_m']
    metal = models['metal']
    samples = []
    for index in sorted({0, count // 2, count - 1}):
        vertices = panels[index * 6:(index + 1) * 6]
        corners = {tuple(v[:3]) for v in vertices}
        supported = 0
        for x, y, z in corners:
            found = any(bounds[0][1] <= roof + .13 and bounds[1][1] >= y - .08
                        and max(abs(bounds[j][0] - x) for j in (0, 1)) < .12
                        and max(abs(bounds[j][2] - z) for j in (0, 1)) < .12
                        and bounds[1][1] - bounds[0][1] > .14
                        for _, bounds in metal.triangles)
            supported += found
        require(supported >= 2, f'{building["id"]}: sampled panel {index} has no actual vertical support geometry')
        samples.append({'panel': index, 'supported_corners': supported})
    return {'panels': count, 'support_samples': samples}


def check_balconies(building, models, osm):
    bays = building['balcony_bays']
    require(isinstance(bays, list) and bays, f'{building["id"]}: no per-bay recess metadata')
    samples = []
    # 跨开间和楼层取六个房间中部，避免用阳台中柱位置误判封死。
    indexes = sorted({round((len(bays) - 1) * i / 5) for i in range(6)})
    for index in indexes:
        bay = bays[index]
        point = finite(bay['clearance_sample_xz_m'], 2, 'balcony sample')
        normal = finite(bay['outward_xz'], 2, 'balcony normal')
        require(abs(math.hypot(*normal) - 1) < .01 and .95 < bay['recess_m'] < 3,
                f'{building["id"]}: invalid recess or outward vector')
        floor = bay['floor_y_m']
        require(building['base_y_m'] < floor < building['base_y_m'] + building['height_m'],
                f'{building["id"]}: balcony floor is outside its model')
        def position(depth, up=1.55):
            return (point[0] + normal[0] * depth, floor + up, point[1] + normal[1] * depth)
        clear_end = -bay['recess_m'] + .20
        hit = first_hit(models, position(.80), position(clear_end))
        require(hit is None, f'{building["id"]}: balcony {index} is blocked before its recessed window: {hit}')
        rear = models['glass'].first_hit(position(clear_end), position(-bay['recess_m'] - .30))
        require(rear is not None, f'{building["id"]}: balcony {index} has no actual recessed rear glazing')
        floor_point = position(-.35, 0)
        floor_hits = surface_hits(models['slab'].vertices, floor_point[0], floor_point[2])
        require(any(abs(height - floor) < .04 for height in floor_hits),
                f'{building["id"]}: balcony {index} has no actual floor slab')
        require(osm.covers(building['osm'], (point[0] - normal[0] * .65, point[1] - normal[1] * .65)),
                f'{building["id"]}: balcony recess points away from its own footprint')
        samples.append({'bay': index, 'floor': bay['floor'], 'clear_inside_depth_m': -clear_end,
                        'rear_glass_xyz_m': rear})
    return samples


def check_qiaoxiang_entry(building, models, osm):
    lines = building['entry_paths_xz_m']
    require(lines and building['entry_clear_height_m'] > 1.3, 'qiaoxiang: missing entrance path or clearance')
    y = building['base_y_m'] + 1.3
    # 仅保留人体高度附近的面，避免对整座高楼做重复的大量三角面检测。
    triangles = [(role, tri, bounds) for role, model in models.items() for tri, bounds in model.triangles
                 if bounds[0][1] - 1e-5 <= y <= bounds[1][1] + 1e-5]
    checked, enters = 0, False
    for line in lines:
        require(len(line) >= 2, 'qiaoxiang: empty entrance polyline')
        for a, b in zip(line, line[1:]):
            finite(a, 2, 'entrance point');finite(b, 2, 'entrance point')
            length = math.dist(a, b)
            if length < .001:
                continue
            normal = (-(b[1] - a[1]) / length, (b[0] - a[0]) / length)
            pieces = max(1, math.ceil(length))
            for i in range(pieces):
                for offset in (-.5, 0, .5):
                    ends = [(a[0] + (b[0] - a[0]) * t + offset * normal[0], y,
                             a[1] + (b[1] - a[1]) * t + offset * normal[1]) for t in (i / pieces, (i + 1) / pieces)]
                    enters |= any(osm.covers(building['osm'], (p[0], p[2])) for p in ends)
                    for role, tri, bounds in triangles:
                        if any(bounds[0][axis] > max(p[axis] for p in ends) + 1e-5
                               or bounds[1][axis] < min(p[axis] for p in ends) - 1e-5 for axis in (0, 2)):
                            continue
                        hit = segment_triangle(ends[0], ends[1], tri)
                        require(hit is None, f'qiaoxiang: garden entrance at base+1.3 m is blocked by {role} at {hit}')
                    checked += 1
    require(enters and checked > 0, 'qiaoxiang: declared garden route does not enter its own footprint')
    return {'segments_at_body_height': checked, 'height_above_model_base_m': 1.3, 'lateral_offsets_m': [-.5, 0, .5]}


def verify(assets, data, groups, require_draped=False):
    osm = OsmSnapshot(data / 'yuehai.osm')
    places = unique(json.loads((data / 'campus-places.json').read_text(encoding='utf-8'))['places'], 'places')
    terrain = Terrain(assets)
    result = {'errors': [], 'buildings': {}, 'limits': [
        'Model/evidence consistency only: nominal heights and details are not claimed as field measurements.',
        'Balcony and rooftop support rays are sampled; no GUI, texture OCR, full route or autumn-dorm road clearance check.',
    ]}
    owners, checked_textures, bridge = {}, set(), None
    for group in groups:
        try:
            manifest, buildings, configs, catalog = validate_manifest(assets, data, group, owners, checked_textures)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['errors'].append({'building': group, 'check': 'manifest', 'detail': str(error)})
            continue
        for ident, building in buildings.items():
            try:
                binding = check_binding(building, configs[ident], osm, places, data)
                if group == 'qiaoyuan' and require_draped:
                    require(qiaoyuan_walking_mode(building) == 'terrain_draped',
                            f'{ident}: final assets still use the old flat entrance mode')
                models, degenerate, elevated_meshes, skywalk_contexts, skywalk_metrics = {}, 0, {}, [], []
                for layer in building['layers']:
                    mesh = Mesh(assets / layer['mesh'])
                    require(len(mesh.vertices) == layer['vertices'], f'{ident}: wrong vertex count {mesh.name}')
                    degenerate += mesh_quality(mesh)
                    models[layer['role']] = mesh
                walking = Mesh(assets / building['walking_mesh']) if building.get('walking_mesh') else None
                require(walking is not None or not building.get('entry_paths_xz_m'), f'{ident}: entrance paths have no walking resource')
                if walking:
                    degenerate += mesh_quality(walking)
                    require(len(walking.vertices) == building['walking_vertices'], f'{ident}: wrong walking vertex count')
                if group == 'south' and ident in SOUTH_SKYWALK_ENDPOINTS:
                    if bridge is None:
                        bridge = Mesh(resource_path(assets, 'campus-bridge.mesh'))
                    registry = unique(manifest['walking_meshes'], 'elevated walking registry')
                    for interface in building['skywalks']:
                        elevated = Mesh(assets / interface['walking_mesh'])
                        degenerate += mesh_quality(elevated)
                        require(len(elevated.vertices) == registry[interface['id']]['vertices'],
                                f'{ident}: elevated walking vertex count disagrees with the actual file')
                        elevated_meshes[interface['walking_mesh']] = elevated
                        context, sky_metrics = check_south_skywalk(building, catalog, registry[interface['id']],
                                                                  elevated, models, osm, bridge)
                        skywalk_contexts.append(context)
                        skywalk_metrics.append(sky_metrics)
                metrics = check_extent(building, configs[ident], models, walking, osm, terrain, group, skywalk_contexts)
                metrics.update({'group': group, 'name_binding': binding, 'layers': len(models),
                                'vertices': sum(len(m.vertices) for m in models.values()), 'degenerate_triangles_excluded': degenerate,
                                'rooftop_panels': check_solar(building, configs[ident], models)})
                if skywalk_metrics:
                    metrics['skywalk_interfaces'] = skywalk_metrics
                if group == 'qiaoyuan':
                    metrics['balcony_samples'] = check_balconies(building, models, osm)
                    if qiaoyuan_walking_mode(building) == 'terrain_draped':
                        metrics['walking_surface'] = check_draped_walk(building, models, walking, terrain)
                        metrics['entry_clearance'] = check_grounded_entry(building, models, osm, terrain)
                    else:
                        metrics['walking_surface'] = {'mode': 'flat', 'expected_plane_y_m': building['base_y_m'] + .035}
                        if ident == 'qiaoxiang':
                            metrics['garden_entry'] = check_qiaoxiang_entry(building, models, osm)
                result['buildings'][ident] = metrics
                mode = '; entrance=' + metrics['walking_surface']['mode'] if group == 'qiaoyuan' else ''
                print(f'PASS: {ident}: binding, independent L/W files, geometry, footprint and height metadata{mode}.', flush=True)
            except (OSError, ValueError, KeyError, TypeError) as error:
                result['errors'].append({'building': ident, 'check': 'geometry', 'detail': str(error)})
            finally:
                # 每栋释放网格，避免把十一栋细栏杆的所有 Python 三角面同时留在内存。
                models, walking, elevated_meshes, skywalk_contexts = {}, None, {}, []
    result['passed'] = not result['errors']
    if 'south' in groups:
        checked = [interface for building in result['buildings'].values() for interface in building.get('skywalk_interfaces', [])]
        result['independent_south_bridge_audit'] = {'required_interfaces': 3, 'verified_interfaces': len(checked),
                                                  'verified_original_body_rays': sum(len(item['audit_rays']) for item in checked)}
        require(not result['passed'] or len(checked) == 3, 'All three independently mapped bridge entrances must be verified')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--group', choices=('all', *GROUP_IDS), default='all')
    parser.add_argument('--require-terrain-draped', action='store_true',
                        help='Reject stale flat Qiaoyuan assets when accepting the new entrance surfaces')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.assets, args.data, list(GROUP_IDS) if args.group == 'all' else [args.group],
                        require_draped=args.require_terrain_draped)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'passed': False, 'errors': [{'building': 'all', 'check': 'input', 'detail': str(error)}]}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for error in result['errors']:
        print(f'FAIL: {error["building"]}/{error["check"]}: {error["detail"]}', file=sys.stderr)
    for limit in result.get('limits', []):
        print('LIMIT: ' + limit)
    print('Dorm asset verification ' + ('passed.' if result['passed'] else 'failed.'))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
