"""Independent M0 checks against actual GLB and legacy geometry, not exporter logs."""
import hashlib
import json
import math
from itertools import combinations
import struct
from pathlib import Path

IDENTITY = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or path == root or root not in path.parents:
        raise ValueError(f'Path must stay inside package: {relative}')
    return path


def multiply(a, b):
    # Column-major matrices, column vectors, matching glTF, Three.js and GLM.
    return [sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
            for col in range(4) for row in range(4)]


def point(matrix, value):
    return [sum(matrix[k * 4 + row] * value[k] for k in range(3)) + matrix[12 + row]
            for row in range(3)]


def node_matrix(node):
    if 'matrix' in node:
        return node['matrix']
    x, y, z, w = node.get('rotation', [0, 0, 0, 1])
    sx, sy, sz = node.get('scale', [1, 1, 1])
    tx, ty, tz = node.get('translation', [0, 0, 0])
    return [(1-2*y*y-2*z*z)*sx, (2*x*y+2*z*w)*sx, (2*x*z-2*y*w)*sx, 0,
            (2*x*y-2*z*w)*sy, (1-2*x*x-2*z*z)*sy, (2*y*z+2*x*w)*sy, 0,
            (2*x*z+2*y*w)*sz, (2*y*z-2*x*w)*sz, (1-2*x*x-2*y*y)*sz, 0,
            tx, ty, tz, 1]


def worlds(nodes):
    result, visiting = {}, set()
    def visit(key):
        if key in result:
            return result[key]
        if key in visiting:
            raise ValueError(f'Hierarchy cycle: {key}')
        visiting.add(key)
        n = nodes[key]
        parent = n['parent']
        result[key] = multiply(visit(parent), n['local_matrix']) if parent else n['local_matrix']
        visiting.remove(key)
        return result[key]
    for key in nodes:
        visit(key)
    return result


def near(actual, expected, label, tolerance=1e-4):
    if len(actual) != len(expected) or max(abs(a-b) for a, b in zip(actual, expected)) > tolerance:
        raise ValueError(f'{label}: actual={actual}, expected={expected}')


def bounds(points):
    low, high, count = [math.inf]*3, [-math.inf]*3, 0
    for p in points:
        if not all(math.isfinite(x) for x in p):
            raise ValueError('Non-finite geometry')
        count += 1
        low = [min(a, b) for a, b in zip(low, p)]
        high = [max(a, b) for a, b in zip(high, p)]
    if not count:
        raise ValueError('Empty geometry')
    return {'min': low, 'max': high}


def read_glb(path):
    data = path.read_bytes()
    if len(data) < 20 or struct.unpack_from('<III', data) != (0x46546c67, 2, len(data)):
        raise ValueError('Invalid GLB header')
    chunks, offset = {}, 12
    while offset < len(data):
        size, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        if offset + size > len(data):
            raise ValueError('Truncated GLB chunk')
        chunks[kind] = data[offset:offset+size]
        offset += size
    doc = json.loads(chunks[0x4e4f534a])
    binary = chunks[0x004e4942]
    if any('uri' in x for x in doc.get('buffers', []) + doc.get('images', [])):
        raise ValueError('M0 GLB must embed buffers and images')
    return doc, binary


def check_legacy_tables(package, manifest):
    """Compare the actual tables consumed by C++, not just their manifest description."""
    def rows(name):
        return [line.split('\t') for line in (package/'legacy'/name).read_text(encoding='utf-8').splitlines()
                if line and not line.startswith('#')]
    node_rows = rows('nodes.tsv')
    indexed = {r[1]: r for r in node_rows}
    if len(indexed) != len(node_rows) or set(indexed) != {n['id'] for n in manifest['nodes']}:
        raise ValueError('Legacy node identities disagree with manifest')
    for n in manifest['nodes']:
        r = indexed[n['id']]
        if len(r) != 20 or r[0] != 'N' or r[2] != (n['parent'] or '-') or set(r[3].split(',')) != set(n['roles']):
            raise ValueError('Legacy node hierarchy/roles disagree with manifest')
        near(list(map(float, r[4:])), n['local_matrix'], 'Legacy local matrix', 1e-6)
    actual = {tuple(r) for r in rows('primitives.tsv')}
    expected = {('P', p['id'], p['node'], p['material'], p['mesh']) for p in manifest['primitives']}
    if actual != expected: raise ValueError('Legacy primitive bindings disagree with manifest')
    material_rows = {r[1]: r for r in rows('materials.tsv')}
    if set(material_rows) != {m['id'] for m in manifest['materials']}:
        raise ValueError('Legacy material identities disagree with manifest')
    for m in manifest['materials']:
        r = material_rows[m['id']]
        if len(r) != 10 or r[:4] != ['T', m['id'], m['legacy_texture'] or '-', m['alpha_mode']] or int(r[5]) != int(m['double_sided']):
            raise ValueError('Legacy material mapping mismatch')
        near([float(r[4]), *map(float, r[6:])], [m['alpha_cutoff'], *m['base_color']], 'Legacy material values')
    anchor_rows = {r[1]: r for r in rows('anchors.tsv')}
    if set(anchor_rows) != {a['id'] for a in manifest['anchors']}:
        raise ValueError('Legacy anchor identities disagree with manifest')
    for a in manifest['anchors']:
        r = anchor_rows[a['id']]
        if r[:3] != ['A', a['id'], a['node']]: raise ValueError('Legacy anchor node mismatch')
        near(list(map(float, r[3:])), [*a['local'], *a['world']], 'Legacy anchor values')
    control_rows = rows('controls.tsv')
    interaction = manifest['interaction']; hierarchy = manifest['hierarchy_probe']
    expected_controls = [['I', interaction['node'], *interaction['axis'], *interaction['range_deg']],
                         ['H', hierarchy['parent'], hierarchy['descendant'], hierarchy['rotation_deg']]]
    expected_controls += [['V', name, *view['eye'], *view['target']] for name, view in manifest['views'].items()]
    if len(control_rows) != len(expected_controls): raise ValueError('Legacy controls count mismatch')
    for expected in expected_controls:
        matches = [r for r in control_rows if r[:2] == expected[:2]]
        if len(matches) != 1 or len(matches[0]) != len(expected): raise ValueError('Legacy control identity mismatch')
        for actual, wanted in zip(matches[0], expected):
            if isinstance(wanted, str):
                if actual != wanted: raise ValueError('Legacy control mapping mismatch')
            else: near([float(actual)], [wanted], 'Legacy control number')


def check_package(package, schema_dir):
    from jsonschema import Draft202012Validator
    manifest = json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    schema = json.loads((schema_dir/'manifest.schema.json').read_text(encoding='utf-8'))
    Draft202012Validator(schema).validate(manifest)
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Non-finite manifest value')
        if isinstance(value, dict):
            for child in value.values(): finite(child)
        if isinstance(value, list):
            for child in value: finite(child)
    finite(manifest)
    for name, expected in manifest['files'].items():
        path = inside(package, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Output hash mismatch: {name}')
    required = {manifest['model'], 'model.blend', *['legacy/'+n+'.tsv' for n in ['nodes','materials','primitives','anchors','controls']]}
    required.update(p['mesh'] for p in manifest['primitives'])
    required.update(m['legacy_texture'] for m in manifest['materials'] if m['legacy_texture'])
    if not required.issubset(manifest['files']): raise ValueError('Unfingerprinted runtime resources')
    check_legacy_tables(package, manifest)
    ns = {n['id']: n for n in manifest['nodes']}
    if len(ns) != len(manifest['nodes']):
        raise ValueError('Duplicate node IDs')
    world = worlds(ns)
    depth = 0
    for key in ns:
        chain = 1
        while ns[key]['parent']:
            key = ns[key]['parent']; chain += 1
        depth = max(depth, chain)
    minimum_depth = manifest.get('validation_requirements', {}).get('min_hierarchy_depth', 4)
    if depth < minimum_depth:
        raise ValueError(f'Expected a real {minimum_depth}-level hierarchy')
    mats = {m['id']: m for m in manifest['materials']}
    positions = []
    triangle_count = 0
    for primitive in manifest['primitives']:
        if primitive['material'] not in mats:
            raise ValueError('Unknown material')
        data = inside(package, primitive['mesh']).read_bytes()
        count, = struct.unpack_from('<I', data, 4)
        if data[:4] != b'SZM1' or count % 3 or len(data) != 8+count*32:
            raise ValueError(f'Invalid SZM1: {primitive["mesh"]}')
        triangle_count += count//3
        for vertex in struct.iter_unpack('<8f', data[8:]):
            if not all(math.isfinite(x) for x in vertex): raise ValueError('Invalid vertex')
            positions.append(point(world[primitive['node']], vertex[:3]))
    legacy_bounds = bounds(positions)
    for side in ('min', 'max'):
        near(legacy_bounds[side], manifest['bounds'][side], 'Legacy bounds')
    if triangle_count != manifest['statistics']['triangles']:
        raise ValueError('Triangle count mismatch')
    doc, binary = read_glb(inside(package, manifest['model']))
    if manifest.get('preview_context'):
        preview = manifest['preview_context']
        if preview.get('role') != 'read_only_surroundings' or preview['model'] not in manifest['files']:
            raise ValueError('Preview context must be separate, read-only and fingerprinted')
        context_doc, _ = read_glb(inside(package, preview['model']))
        if set(ns).intersection(n.get('name') for n in context_doc.get('nodes', [])):
            raise ValueError('Context reuses editable asset node identities')
    glb_nodes = {}
    for i, n in enumerate(doc.get('nodes', [])):
        glb_nodes[str(i)] = {'parent': None, 'local_matrix': node_matrix(n)}
    for i, n in enumerate(doc.get('nodes', [])):
        for child in n.get('children', []): glb_nodes[str(child)]['parent'] = str(i)
    glb_world = worlds(glb_nodes)
    named = {n.get('name'): glb_world[str(i)] for i, n in enumerate(doc['nodes'])}
    for key, transform in world.items():
        if key not in named: raise ValueError(f'GLB lost named node: {key}')
        near(named[key], transform, f'GLB transform {key}')
    glb_positions = []
    for i, n in enumerate(doc['nodes']):
        if 'mesh' not in n: continue
        for primitive in doc['meshes'][n['mesh']]['primitives']:
            accessor = doc['accessors'][primitive['attributes']['POSITION']]
            if accessor['componentType'] != 5126 or accessor['type'] != 'VEC3' or 'sparse' in accessor:
                raise ValueError('Unsupported M0 POSITION accessor')
            view = doc['bufferViews'][accessor['bufferView']]
            start = view.get('byteOffset', 0)+accessor.get('byteOffset', 0)
            for vertex in range(accessor['count']):
                p = struct.unpack_from('<3f', binary, start+vertex*view.get('byteStride', 12))
                glb_positions.append(point(glb_world[str(i)], p))
    glb_bounds = bounds(glb_positions)
    for side in ('min', 'max'): near(glb_bounds[side], legacy_bounds[side], 'Cross-export bounds')
    for anchor in manifest['anchors']:
        near(point(world[anchor['node']], anchor['local']), anchor['world'], 'Legacy anchor')
        near(point(named[anchor['node']], anchor['local']), anchor['world'], 'GLB anchor')
    independent = False
    for a, b, c in combinations([v['world'] for v in manifest['anchors']], 3):
        u, v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
        cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        if sum(x*x for x in cross) >= 1e-8:
            independent = True; break
    if not independent: raise ValueError('Collinear anchors')
    mask_materials = [m for m in mats.values() if m['alpha_mode'] == 'MASK']
    mask_required = manifest.get('validation_requirements', {'mask': True}).get('mask', False)
    if (mask_required or mask_materials) and (not mask_materials or not any(m.get('alphaMode') == 'MASK' for m in doc['materials'])):
        raise ValueError('Missing actual MASK material')
    for material in mask_materials:
        data = inside(package, material['legacy_texture']).read_bytes()
        w, h = struct.unpack_from('<II', data)
        if len(data) != 8+w*h*4: raise ValueError('Invalid RGBA texture size')
        alpha = data[11::4]
        if not min(alpha) < material['alpha_cutoff']*255 < max(alpha):
            raise ValueError('MASK texture does not contain both opaque and transparent pixels')
    return {'status': 'passed', 'triangles': triangle_count, 'nodes': len(ns),
            'hierarchy_depth': depth, 'legacy_bounds': legacy_bounds, 'glb_bounds': glb_bounds,
            'anchors_checked': len(manifest['anchors']), 'manifest_sha256': sha256(package/'manifest.json')}
