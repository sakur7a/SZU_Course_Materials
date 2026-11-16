"""Blender background exporter. Arguments after -- are layout JSON, assets, output."""
import json
import math
import random
import struct
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree


X0, Z0 = -332., -49.
CURBS = 'campus-lide-corridor-curbs.mesh'
FOLIAGE = 'campus-lide-corridor-leaves.mesh'


def mesh_object(name, vertices, faces, collection, material, *, recalculate=False):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    if recalculate:
        data = bmesh.new()
        data.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(data, faces=list(data.faces))
        data.to_mesh(mesh)
        data.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def collection(name):
    group = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(group)
    return group


def material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1.)
    return mat


def context_mesh(path, group, mat):
    raw = path.read_bytes()
    count, = struct.unpack_from('<I', raw, 4)
    if raw[:4] != b'SZM1' or count % 3 or len(raw) != 8 + count * 32:
        raise ValueError(f'Invalid context mesh: {path}')
    points = np.frombuffer(raw, '<f4', offset=8).reshape(-1, 3, 8)
    x, z = points[:, :, 0], points[:, :, 2]
    keep = (x.max(1) >= -435) & (x.min(1) <= -208) & (z.max(1) >= -86) & (z.min(1) <= -15)
    vertices = [(x-X0, Z0-z, y) for x, y, z, *_ in points[keep].reshape(-1, 8)]
    faces = [(i, i+1, i+2) for i in range(0, len(vertices), 3)]
    return mesh_object(path.stem, vertices, faces, group, mat)


def sampled_line(points, spacing):
    # Keep input corners, then add stations to follow the rendered slope.
    result = [Vector(points[0])]
    for point in points[1:]:
        start, end = result[-1], Vector(point)
        length = (end-start).length
        if length < .008:
            continue
        steps = math.ceil(length / spacing)
        result.extend(start.lerp(end, step/steps) for step in range(1, steps+1))
    return result


def sweep(name, points, profile, ground, group, mat, *, closed=False, soft=False):
    stations = sampled_line(points, .48 if not soft else .54)
    if closed:
        stations.pop()
    if len(stations) < 2:
        raise ValueError(f'Invalid profile sweep: {name}')
    vertices, faces = [], []
    for i, p in enumerate(stations):
        before = stations[i-1] if closed or i > 0 else p
        after = stations[(i+1) % len(stations)] if closed or i+1 < len(stations) else p
        tangent = (after-before).normalized()
        normal = Vector((-tangent.y, tangent.x))
        if closed:
            taper = 1.
        else:
            end_dist = min((p-stations[0]).length, (p-stations[-1]).length)
            taper = min(1., .30 + end_dist/.58)
        # Geometry stays low at open ends. Each edge samples the ground separately.
        wave = 1. + (.06*math.sin(i*1.7) if soft else 0.)
        for offset, height in profile:
            x, z = p + offset * normal
            vertices.append((x-X0, Z0-z, ground(x, z) + height*taper*wave))
    n = len(profile)
    for i in range(len(stations) if closed else len(stations)-1):
        j = (i+1) % len(stations)
        for k in range(n):
            l = (k+1) % n
            faces.append((i*n+k, i*n+l, j*n+l, j*n+k))
    if not closed:
        faces.extend((tuple(reversed(range(n))), tuple(range(len(vertices)-n, len(vertices)))))
    obj = mesh_object(name, vertices, faces, group, mat, recalculate=True)
    uv = obj.data.uv_layers.new()
    for poly in obj.data.polygons:
        poly.use_smooth = soft
        for i, loop in enumerate(poly.loop_indices):
            x, y, z = vertices[obj.data.loops[loop].vertex_index]
            if soft:
                # Use the dense part of the leaf atlas for the opaque hedge body.
                uv.data[loop].uv = ((.27,.31),(.44,.31),(.44,.47),(.27,.47))[i % 4]
            else:
                uv.data[loop].uv = (x*.72+z*.26, -y*.72+z*.28)
    return obj


def leaf_cards(name, points, half_width, height, ground, group, mat, seed):
    rng = random.Random(seed)
    vertices, faces = [], []
    stations = sampled_line(points, .42)
    for a, b in zip(stations, stations[1:]):
        if (b-a).length < .07:
            continue
        direction = (b-a).normalized()
        normal = Vector((-direction.y, direction.x))
        for side in (-1., 1.):
            p = a.lerp(b, .5) + side * half_width * rng.uniform(.33, .56) * normal
            width = min(.20, half_width*.83) * rng.uniform(.84, 1.12)
            depth = half_width * .48
            corners = [p + w*direction + d*normal
                       for w, d in ((-width, -depth), (width, -depth), (width, depth), (-width, depth))]
            base = len(vertices)
            for index, (x, z) in enumerate(corners):
                bump = (-.055 if index < 2 else .03) * side
                vertices.append((x-X0, Z0-z, ground(x, z)+height*rng.uniform(.84,.98)+bump))
            faces.append((base, base+1, base+2, base+3))
    obj = mesh_object(name, vertices, faces, group, mat)
    uv = obj.data.uv_layers.new()
    for face in obj.data.polygons:
        for i, loop in enumerate(face.loop_indices):
            uv.data[loop].uv = ((0,0),(1,0),(1,1),(0,1))[i]
    return obj


def gate_shrub_clusters(lines, ground, group, mat):
    """Build overlapping lobes inside the photo-estimated gate bed."""
    rng = random.Random(20261106)
    source = bmesh.new()
    bmesh.ops.create_icosphere(source, subdivisions=2, radius=1)
    source.verts.index_update()
    unit_vertices = [v.co.copy() for v in source.verts]
    unit_faces = [tuple(v.index for v in f.verts) for f in source.faces]
    source.free()
    for row, line in enumerate(lines):
        for index, point in enumerate(sampled_line(line, .62)):
            if row and index % 4 == 0:
                continue
            vertices, faces = [], []
            height = rng.uniform(.36, .65)
            # The larger lobe anchors to the ground. Two smaller lobes break up the crown.
            for dx, dz, radius, scale in ((0.,0.,.325,1.),
                                          (.12,-.10,.20,.83),(-.12,.05,.19,.74)):
                x, z = point + Vector((dx, dz))
                y = ground(x, z)
                base = len(vertices)
                for u, v, w in unit_vertices:
                    vertices.append((x-X0 + radius*u, Z0-z + radius*v*.88,
                                     y + height*(.47 + .48*w)*scale))
                faces.extend(tuple(base+k for k in face) for face in unit_faces)
            obj = mesh_object(f'Gate bed crown {row+1}.{index+1}', vertices, faces, group, mat)
            uv = obj.data.uv_layers.new()
            for face in obj.data.polygons:
                face.use_smooth = True
                for i, loop in enumerate(face.loop_indices):
                    uv.data[loop].uv = ((.273,.314),(.441,.326),(.361,.468))[i]


def export(group, path):
    payload = bytearray()
    for obj in group.objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        for face in mesh.loop_triangles:
            if face.area < 1e-8:
                continue
            for loop in face.loops:
                x, y, z = mesh.vertices[mesh.loops[loop].vertex_index].co
                normal = mesh.corner_normals[loop].vector
                if obj.get('leaf_cards'):
                    normal = Vector((normal.x*.33, normal.y*.33, abs(normal.z)+.55)).normalized()
                nx, ny, nz = normal
                u, v = mesh.uv_layers.active.data[loop].uv
                payload.extend(struct.pack('<8f', x+X0, z, Z0-y, nx, nz, -ny, u, v))
    if not payload or len(payload)//32 > 500000:
        raise ValueError(f'Unsupported mesh size for {path}')
    path.write_bytes(b'SZM1' + struct.pack('<I', len(payload)//32) + payload)
    return len(payload)//96


def main():
    args = sys.argv[sys.argv.index('--')+1:]
    layout, assets, output = map(Path, args)
    config = json.loads(layout.read_text(encoding='utf-8'))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    reference = collection('Existing terrain and campus context')
    stone = collection('Editable stone edges and planter rim')
    plants = collection('Editable planting strips and foliage')
    ground_obj = context_mesh(assets/'campus-terrain.mesh', reference, material('Ground', (.32,.44,.22)))
    bvh = BVHTree.FromPolygons([v.co for v in ground_obj.data.vertices],
                              [p.vertices[:] for p in ground_obj.data.polygons], all_triangles=True)

    def ground(x, z):
        position, _, _, _ = bvh.ray_cast((x-X0, Z0-z, 150.), (0.,0.,-1.))
        if position is None:
            raise ValueError(f'No rendered terrain at {(x,z)}')
        return position.z

    for file, color in (
        ('campus-road-surface.mesh', (.24,.25,.25)),
        ('campus-lide-walk.mesh', (.61,.29,.20)),
        ('campus-time-paving.mesh', (.64,.63,.60)),
        ('campus-gate-lide-stone.mesh', (.56,.56,.54)),
        ('campus-gate-lide-roof.mesh', (.53,.53,.52)),
    ):
        context_mesh(assets/file, reference, material(file, color))
    for name, color in (('paving', (.72,.71,.67)), ('lawn', (.30,.52,.17)), ('bed', (.27,.22,.15))):
        context_mesh(layout.parent/f'campus-lide-corridor-{name}.mesh', reference, material(name, color))
    limestone = material('Light gray stone', (.74,.72,.66))
    leafy = material('Leaf atlas', (.31,.49,.17))
    leafy.use_nodes = True
    atlas = (assets/'campus-leaf-cluster.rgba').read_bytes()
    width, height = struct.unpack_from('<II', atlas)
    texture = bpy.data.images.new('Packed campus foliage', width=width, height=height, alpha=True)
    texture.pixels.foreach_set(np.frombuffer(atlas, dtype=np.uint8, offset=8).astype(np.float32)/255.)
    texture.pack()
    nodes = leafy.node_tree.nodes
    sampler = nodes.new('ShaderNodeTexImage')
    sampler.image = texture
    shader = nodes.get('Principled BSDF')
    leafy.node_tree.links.new(sampler.outputs['Color'], shader.inputs['Base Color'])
    leafy.node_tree.links.new(sampler.outputs['Alpha'], shader.inputs['Alpha'])

    def curb_profile(width, top):
        return ((-width,-.04),(-width,top-.028),(-width+.024,top),
                (width-.024,top),(width,top-.028),(width,-.04))

    for i, points in enumerate(config['gate_curbs']):
        sweep(f'Red path stone edge {i+1}', points, curb_profile(.092,.16), ground, stone, limestone)
    for i, points in enumerate(config['lawn_curbs']):
        sweep(f'Lawn side edge {i+1}', points, curb_profile(.086,.13), ground, stone, limestone)
    sweep('Lide planting bed chamfered rim', config['gate_bed'], curb_profile(.083,.155),
          ground, stone, limestone, closed=True)

    for i, points in enumerate(config['hedges']):
        w, h = .32,.34
        section = ((-w,-.023),(-w,h*.51),(-w*.73,h*.88),(-w*.30,h),
                   (w*.30,h),(w*.73,h*.88),(w,h*.51),(w,-.023))
        sweep(f'Low hedge {i+1}', points, section, ground, plants, leafy, soft=True)
        leaves = leaf_cards(f'Canopy detail {i+1}', points, w, h, ground, plants, leafy, 20261106+i)
        leaves['leaf_cards'] = True
    gate_shrub_clusters(config['gate_planting'], ground, plants, leafy)

    curb_count, leaf_count = export(stone, output/CURBS), export(plants, output/FOLIAGE)
    bpy.context.scene.unit_settings.system = 'METRIC'
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                view = area.spaces.active
                view.shading.color_type = 'MATERIAL'
                view.overlay.show_floor = False
                view.overlay.show_axis_x = False
                view.overlay.show_axis_y = False
                view.region_3d.view_location = (-10, 1, 24)
                view.region_3d.view_rotation = Quaternion((1,0,0), .62)
                view.region_3d.view_distance = 183
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'lide-timesquare.blend'))
    print(f'SZM1 exports: {curb_count} stone triangles; {leaf_count} planting triangles')


if __name__ == '__main__':
    main()
