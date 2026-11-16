"""照片与标注驱动的独立植被层；无坐标证据时生成的点必须标记为示意。"""
import json
import math
import random
import struct

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


def _mesh_footprint(path, region):
    """只读取标注附近的地面三角形，避免把整校路网重新做布尔运算。"""
    with path.open('rb') as stream:
        if stream.read(4) != b'SZM1':
            raise ValueError(f'Invalid mesh: {path}')
        count, = struct.unpack('<I', stream.read(4))
        payload = stream.read()
    if count % 3 or len(payload) != count * 32:
        raise ValueError(f'Truncated mesh: {path}')
    vertices = list(struct.iter_unpack('<8f', payload))
    xmin, zmin, xmax, zmax = region.bounds
    triangles = []
    for i in range(0, len(vertices), 3):
        triangle = vertices[i:i+3]
        xs = [v[0] for v in triangle]; zs = [v[2] for v in triangle]
        if max(xs) < xmin or min(xs) > xmax or max(zs) < zmin or min(zs) > zmax:
            continue
        shape = Polygon(zip(xs, zs))
        if shape.is_valid and shape.area > 1e-9 and shape.intersects(region):
            triangles.append(shape.intersection(region))
    return unary_union(triangles)


def generate(*, config, assets, campus, vehicle_surface, buildings, waters, pitches,
             booth, bridge_clearance, height_at, mesh_type):
    """输出独立灌木网格、可实例化树点和记录；原全校随机树不改动。"""
    regions = config.get('regions', [])
    tree_points = config.get('tree_points', [])
    footprints = []
    identities = set()
    for item in regions:
        polygon = Polygon(item['polygon_xz_m'])
        if item['kind'] != 'shrub_cluster' or not polygon.is_valid or polygon.area <= 0:
            raise ValueError(f"Invalid shrub region: {item.get('id')}")
        if (not item.get('id') or item['id'] in identities or not item.get('source') or
                item.get('confidence') not in ('illustrative', 'photo_estimate', 'measured')):
            raise ValueError(f"Missing id/source/confidence or duplicate id: {item.get('id')}")
        identities.add(item['id'])
        if (item['count'] < 0 or item['min_spacing_m'] <= 0 or
                not 0 < item['radius_m'][0] <= item['radius_m'][1] or
                not 0 < item['height_m'][0] <= item['height_m'][1]):
            raise ValueError(f"Invalid shrub parameters: {item['id']}")
        footprints.append(polygon)
    footprints += [Point(*item['center_xz_m']).buffer(item.get('clearance_m', 3.0))
                   for item in tree_points]
    study = unary_union(footprints).buffer(2.0).envelope if footprints else box(0,0,0,0)
    pedestrian = unary_union([_mesh_footprint(assets/name, study) for name in (
        'campus-network-sidewalk.mesh','campus-network-paths.mesh',
        'campus-lide-walk.mesh','campus-lide-walk-edge.mesh')])
    structures = unary_union([g for _,_,g in buildings if g.intersects(study)] + [booth])
    hazard = unary_union([vehicle_surface.intersection(study), pedestrian, structures,
                          bridge_clearance.intersection(study),
                          *[g for _,_,g in waters if g.intersects(study)],
                          *[g for _,_,g in pitches if g.intersects(study)]])
    forbidden = hazard.buffer(.30)
    with (assets/'campus-trees.bin').open('rb') as stream:
        count, = struct.unpack('<I',stream.read(4))
        existing = list(struct.iter_unpack('<4f',stream.read()))
    if len(existing) != count:
        raise ValueError('Truncated baseline trees')
    expanded_study=study.buffer(2)
    local_trees=[t for t in existing if expanded_study.covers(Point(t[0],t[2]))]
    shrub_mesh = mesh_type()
    shrubs, accepted_trees, rejected = [], [], []
    for item in regions:
        zone = Polygon(item['polygon_xz_m'])
        rng = random.Random(item['seed'])
        xmin,zmin,xmax,zmax = zone.bounds
        selected = []
        for _ in range(max(200, item['count']*100)):
            if len(selected) >= item['count']:
                break
            x,z = rng.uniform(xmin,xmax),rng.uniform(zmin,zmax)
            radius = rng.uniform(*item['radius_m']); height = rng.uniform(*item['height_m'])
            shape = Point(x,z).buffer(radius)
            if not campus.covers(shape) or not zone.covers(shape) or forbidden.intersects(shape):
                continue
            if any(math.hypot(x-t[0],z-t[2]) < radius+.5 for t in local_trees):
                continue
            if any(math.hypot(x-a['x'],z-a['z']) < max(item['min_spacing_m'],radius+a['radius'])
                   for a in shrubs+selected):
                continue
            selected.append({'id':item['id'], 'x':x, 'z':z, 'radius':radius, 'height':height,
                             'confidence':item['confidence'], 'source':item['source']})
        if len(selected) < item['count']:
            rejected.append({'id':item['id'],'reason':f"placed {len(selected)} of {item['count']} after clearance"})
        shrubs.extend(selected)
    for item in shrubs:
        x,z,r,h = item['x'],item['z'],item['radius'],item['height']
        for angle in (0,math.pi/3,2*math.pi/3):
            dx,dz=math.cos(angle)*r,math.sin(angle)*r
            left_y=height_at(x-dx,z-dz)+.03; right_y=height_at(x+dx,z+dz)+.03
            shrub_mesh.quad((x-dx,left_y,z-dz),(x+dx,right_y,z+dz),
                            (x+dx,right_y+h,z+dz),(x-dx,left_y+h,z-dz),
                            (-math.sin(angle),0,math.cos(angle)),((0,.15),(1,.15),(1,.85),(0,.85)))
    # 树点使用现有宽叶树实例网格；严格过滤已有树、桥、校门、道路及人行空间。
    gates = json.loads((assets/'campus-gates.json').read_text(encoding='utf8'))
    gate_areas = unary_union([box(g['bounds_xyz_m'][0][0],g['bounds_xyz_m'][0][2],
                                  g['bounds_xyz_m'][1][0],g['bounds_xyz_m'][1][2]).buffer(2)
                             for g in gates])
    accepted_ids=[]
    for item in tree_points:
        if (not item.get('id') or item['id'] in identities or item.get('asset') != 'existing_broadleaf' or not item.get('source') or
                item.get('confidence') not in ('illustrative','photo_estimate','measured')):
            raise ValueError(f"Unsupported tree annotation: {item.get('id')}")
        identities.add(item['id'])
        x,z=item['center_xz_m']; clearance=item.get('clearance_m',3.0)
        if not all(math.isfinite(v) for v in (x,z,clearance,item.get('scale',1.0))) or clearance<=0 or item.get('scale',1.0)<=0:
            raise ValueError(f"Invalid tree coordinates or scale: {item['id']}")
        shape=Point(x,z).buffer(clearance)
        reason=None
        if not campus.covers(shape): reason='outside campus'
        elif forbidden.intersects(shape) or gate_areas.intersects(shape) or bridge_clearance.intersects(shape): reason='road, path, gate, bridge or structure clearance'
        elif any(math.hypot(x-t[0],z-t[2]) < clearance+2 for t in existing+accepted_trees): reason='existing tree clearance'
        elif any(math.hypot(x-s['x'],z-s['z']) < clearance+s['radius'] for s in shrubs): reason='annotated shrub clearance'
        if reason:
            rejected.append({'id':item['id'],'reason':reason})
            continue
        accepted_trees.append((x,height_at(x,z),z,item.get('scale',1.0)))
        accepted_ids.append(item['id'])
    if len(existing)+len(accepted_trees)>10000:
        raise ValueError('Too many tree instances for the renderer')
    with (assets/'campus-annotated-trees.bin').open('wb') as stream:
        stream.write(struct.pack('<I',len(accepted_trees)))
        for tree in accepted_trees: stream.write(struct.pack('<4f',*tree))
    if shrubs:
        vertices=shrub_mesh.save(assets/'campus-annotated-shrubs.mesh')
    else:
        # 可关闭所有标注植被；渲染器明确允许此独立图层为空。
        (assets/'campus-annotated-shrubs.mesh').write_bytes(b'SZM1'+struct.pack('<I',0))
        vertices=0
    result={'shrub_vertices':vertices,'shrub_points':shrubs,'tree_instances':len(accepted_trees),
            'accepted_tree_ids':accepted_ids,
            'tree_annotations':tree_points,'rejected':rejected,
            'note':'估计与示意位置不能作为实测；来源及置信度见 data/planting-annotations.json'}
    (assets/'campus-planting-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result
