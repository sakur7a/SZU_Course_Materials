"""Build the independently editable Lide Gate approximation in Blender 4.5.

    blender --background --python build.py -- --output DIRECTORY

Blender coordinates: X east, Y north, Z up, in metres. The terrain snapshot is
only read for local ground heights and preview; it is never exported as part of
the gate. The output GLB and evaluated legacy meshes come from the same scene.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import bpy
from mathutils import Vector

ASSET = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[6]
CONTEXT = ASSET / "context"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-flat-preview", action="store_true",
                        help="temporary design preview only when context terrain is unavailable")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


class Terrain:
    """Sample the cropped project's SZM1 terrain, with no extrapolation."""

    def __init__(self, path, allow_flat=False):
        self.available = path.is_file()
        self.cells = {}
        if not self.available:
            if not allow_flat:
                raise FileNotFoundError("Context terrain is required for a grounded build: " + str(path))
            print("WARNING: no context terrain; using temporary flat z=0 for design preview")
            return
        data = path.read_bytes()
        if data[:4] != b"SZM1" or len(data) < 8:
            raise ValueError("Invalid context terrain header: " + str(path))
        count = struct.unpack_from("<I", data, 4)[0]
        if count % 3 or len(data) != 8 + count * 32:
            raise ValueError("Invalid context terrain size: " + str(path))
        vertices = [record[:3] for record in struct.iter_unpack("<8f", data[8:])]
        for i in range(0, count, 3):
            tri = vertices[i:i + 3]  # local RH Y-up: x east, y height, z south
            x0, x1 = min(v[0] for v in tri), max(v[0] for v in tri)
            z0, z1 = min(v[2] for v in tri), max(v[2] for v in tri)
            for ix in range(math.floor(x0 / 5), math.floor(x1 / 5) + 1):
                for iz in range(math.floor(z0 / 5), math.floor(z1 / 5) + 1):
                    self.cells.setdefault((ix, iz), []).append(tri)
        print("Read", count // 3, "context terrain triangles")

    def height(self, east, north):
        if not self.available:
            return 0.0
        x, z = east, -north
        for a, b, c in self.cells.get((math.floor(x / 5), math.floor(z / 5)), []):
            d = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2])
            if abs(d) < 1e-10:
                continue
            u = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / d
            v = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / d
            w = 1.0 - u - v
            if min(u, v, w) >= -1e-6:
                return u * a[1] + v * b[1] + w * c[1]
        raise ValueError(f"Context terrain does not cover ({east:.3f}, {north:.3f}); refusing extrapolation")

    def extrema(self, x, y, width, depth):
        """Exact triangle/rectangle extrema, rather than a sparse point grid."""
        if not self.available:
            return 0.0, 0.0
        xmin, xmax = x - width / 2, x + width / 2
        zmin, zmax = -y - depth / 2, -y + depth / 2
        # Verify corners really belong to the cropped surface; do not extend
        # the nearest sampled height beyond the snapshot boundary.
        for xx in (xmin, xmax):
            for zz in (zmin, zmax):
                self.height(xx, -zz)
        triangles = []
        for ix in range(math.floor(xmin / 5), math.floor(xmax / 5) + 1):
            for iz in range(math.floor(zmin / 5), math.floor(zmax / 5) + 1):
                triangles.extend(self.cells.get((ix, iz), []))

        def clip(poly, axis, bound, keep_greater):
            result = []
            if not poly:
                return result
            for a, b in zip(poly, poly[1:] + poly[:1]):
                inside_a = a[axis] >= bound if keep_greater else a[axis] <= bound
                inside_b = b[axis] >= bound if keep_greater else b[axis] <= bound
                if inside_a:
                    result.append(a)
                if inside_a != inside_b:
                    t = (bound - a[axis]) / (b[axis] - a[axis])
                    result.append(tuple(a[i] + t * (b[i] - a[i]) for i in range(3)))
            return result

        heights = []
        for tri in triangles:
            a, b, c = tri
            signed_area = (b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0])
            if abs(signed_area) < 1e-10:
                continue
            poly = list(tri)
            for axis, bound, greater in ((0, xmin, True), (0, xmax, False),
                                         (2, zmin, True), (2, zmax, False)):
                poly = clip(poly, axis, bound, greater)
            heights.extend(p[1] for p in poly)
        if not heights:
            raise ValueError(f"No terrain intersects footprint at ({x:.3f}, {y:.3f})")
        return min(heights), max(heights)

    def minimum(self, x, y, width, depth):
        # Foundation extensions avoid floating without adding a walkable slab.
        return self.extrema(x, y, width, depth)[0]


def material(name, color, *, image=None, metallic=0.0, roughness=0.72):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = True
    mat.diffuse_color = color
    mat["alpha_mode"] = "OPAQUE"
    mat["alpha_cutoff"] = 0.5
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    if image:
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        mat.node_tree.links.new(tex.outputs["Color"], shader.inputs["Base Color"])
    return mat


def load_sign(filename):
    image = bpy.data.images.load(str(ASSET / "source" / filename), check_existing=False)
    image.name = filename.rsplit(".", 1)[0]
    image.colorspace_settings.name = "sRGB"
    image.pack()
    image.filepath = "//legacy/textures/" + filename
    return image


def empty(name, parent=None, location=(0, 0, 0), roles="visual"):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    obj.location = location
    obj["roles"] = roles
    return obj


def box(name, parent, location, dimensions, mat, bevel=0.0, roles="visual"):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.name = name
    obj.scale = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.parent = parent
    obj.location = location
    obj.data.materials.append(mat)
    obj["roles"] = roles
    if bevel:
        modifier = obj.modifiers.new("edge_chamfer", "BEVEL")
        modifier.width = bevel
        modifier.segments = 2
        obj.modifiers.new("weighted_corners", "WEIGHTED_NORMAL")
    return obj


def mesh_object(name, parent, vertices, faces, mat, uv_coords=None):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    if uv_coords is not None:
        for polygon in mesh.polygons:
            for loop, coord in zip(polygon.loop_indices, uv_coords):
                uv.data[loop].uv = coord
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    obj["roles"] = "visual"
    obj.data.materials.append(mat)
    return obj


def wall(name, parent, x, y, width, depth, height, low, stone, joint):
    """A wall plus actual fine horizontal joints at approximately 0.24 m."""
    assembly = empty(name + "_assembly", parent, (x, y, 0))
    box(name + "_wall", assembly, (0, 0, (height + low) / 2),
        (width, depth, height - low), stone, 0.018)
    verts, faces = [], []

    def quad(points):
        first = len(verts)
        verts.extend(points)
        faces.append(tuple(range(first, first + 4)))

    for i in range(1, math.floor(height / 0.24) + 1):
        z = i * 0.24
        thickness = 0.024
        for side in (-1, 1):
            xx = side * (width / 2 + 0.012)
            yy = side * (depth / 2 + 0.012)
            # Four faces of the wall retain the close-set joint spacing.
            # Materials are double sided for these intentionally thin strips.
            quad([(xx, -depth / 2, z), (xx, depth / 2, z),
                  (xx, depth / 2, z + thickness), (xx, -depth / 2, z + thickness)])
            quad([(-width / 2, yy, z), (width / 2, yy, z),
                  (width / 2, yy, z + thickness), (-width / 2, yy, z + thickness)])
    mesh_object(name + "_horizontal_joints", assembly, verts, faces, joint)
    return assembly


def west_sign(name, parent, x, y, bottom, width, height, mat):
    # The outside is west (-X). UVs read left-to-right when looking east.
    verts = [(x, y + width / 2, bottom), (x, y - width / 2, bottom),
             (x, y - width / 2, bottom + height), (x, y + width / 2, bottom + height)]
    return mesh_object(name, parent, verts, [(0, 1, 2, 3)], mat,
                       [(0, 0), (1, 0), (1, 1), (0, 1)])


def booth(name, parent, x, y, low, base_top, stone, glass, frame, roof):
    """Independent booth shell, deep glass, individually editable window bars."""
    part = empty(name, parent, (x, y, 0))
    width, depth = 7.6, 7.0
    if base_top >= 2.35:
        raise ValueError(name + " terrain leaves insufficient height for the estimated glazing")
    box(name + "_stone_base", part, (0, 0, (base_top + low) / 2),
        (width, depth, base_top - low), stone, 0.018)
    glass_top = 2.76
    glass_height = glass_top - base_top
    glass_middle = (glass_top + base_top) / 2
    box(name + "_glazing", part, (0, 0, glass_middle),
        (width - 0.10, depth - 0.10, glass_height), glass, 0.012)
    box(name + "_upper_lintel", part, (0, 0, 2.88), (width, depth, 0.18), stone, 0.015)
    # Mullions are individual nodes, rather than being baked into the glazing.
    for i, xx in enumerate((-3.72, -1.86, 0.0, 1.86, 3.72)):
        for side in (-1, 1):
            box(f"{name}_mullion_x_{i}_{side + 1}", part, (xx, side * 3.46, glass_middle),
                (0.055, 0.07, glass_height), frame, 0.006)
    for i, yy in enumerate((-3.42, -1.71, 0.0, 1.71, 3.42)):
        for side in (-1, 1):
            box(f"{name}_mullion_y_{i}_{side + 1}", part, (side * 3.76, yy, glass_middle),
                (0.07, 0.055, glass_height), frame, 0.006)
    for i, zz in enumerate((base_top + 0.03, glass_middle, 2.72)):
        for side in (-1, 1):
            box(f"{name}_rail_x_{i}_{side + 1}", part, (0, side * 3.47, zz),
                (7.57, 0.06, 0.052), frame, 0.005)
            box(f"{name}_rail_y_{i}_{side + 1}", part, (side * 3.77, 0, zz),
                (0.06, 6.95, 0.052), frame, 0.005)
    box(name + "_door_handle", part, (-3.82, -1.0, min(base_top + 0.40, 2.30)),
        (0.055, 0.28, 0.045), roof, 0.008)
    return part


def canopy(name, parent, x, y, pale, trim):
    part = empty(name, parent, (x, y, 0))
    box(name + "_slab", part, (0, 0, 3.38), (10.0, 10.5, 0.36), pale, 0.035)
    box(name + "_lower_edge", part, (0, 0, 3.14), (9.84, 10.34, 0.12), trim, 0.014)
    # Faint parapet joints on the upper surface have a restrained metre scale.
    for i, dx in enumerate((-2.5, 0, 2.5)):
        box(f"{name}_top_joint_{i}", part, (dx, 0, 3.565), (0.012, 10.28, 0.006), trim)
    return part


def barrier_base(name, parent, x, y, metal, roof):
    box(name + "_cabinet", parent, (x, y, 0.52), (0.43, 0.42, 0.94), metal, 0.025)
    box(name + "_cap", parent, (x, y, 1.01), (0.49, 0.47, 0.08), roof, 0.015)


def build_scene(terrain):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0

    stone = material("gate_stone", (0.39, 0.41, 0.40, 1), roughness=0.87)
    joint = material("horizontal_joint", (0.19, 0.21, 0.20, 1), roughness=0.92)
    joint.use_backface_culling = False
    roof = material("canopy_concrete", (0.62, 0.64, 0.61, 1), roughness=0.8)
    metal = material("dark_metal", (0.13, 0.16, 0.16, 1), metallic=0.25, roughness=0.45)
    glass = material("deep_glass", (0.075, 0.16, 0.18, 1), metallic=0.12, roughness=0.24)
    white = material("barrier_white", (0.86, 0.88, 0.83, 1), roughness=0.42)
    red = material("barrier_red", (0.65, 0.12, 0.10, 1), roughness=0.47)
    school = material("school_name", (1, 1, 1, 1), image=load_sign("school_name.png"))
    address = material("address_plate", (1, 1, 1, 1), image=load_sign("address.png"))

    root = empty("root")
    site = empty("site", root)
    architecture = empty("gate_architecture", site)

    # Side group origins retain the old gate footprint and sit on their own
    # terrain sample; extensions reach below the lowest sampled corner.
    for side, label in ((1, "north"), (-1, "south")):
        yy = side * 13.0
        base = terrain.height(0.0, yy)
        group = empty(label + "_gate", architecture, (0, yy, base))
        if label == "north":
            specs = (("north_outer", 0.0, 1.8, 4.5, 2.3, 12.4),
                     ("north_inner", 1.2, -1.8, 4.5, 2.3, 11.4))
        else:
            specs = (("south_outer", 0.0, 0.9, 4.5, 4.4, 13.0),
                     ("south_return", 2.1, -2.0, 4.5, 3.0, 12.0))
        for name, x, y, width, depth, height in specs:
            low = terrain.minimum(x, yy + y, width, depth) - base - 0.12
            wall(name, group, x, y, width, depth, height, low, stone, joint)
        if label == "north":
            # Bridge above the opening. Clear space between the twin walls is
            # intentionally preserved below it, as in the photographic view.
            # Fill only the gap between piers: extending a broad box through
            # either wall would create coincident exterior faces and flicker.
            box("north_opening_lintel", group, (0.6, 0, 6.40), (3.28, 1.32, 1.20), stone, 0.015)
        else:
            # Both signs are attached to the west face of the actual wall.
            west_sign("school_name_sign", group, -2.285, 1.50, 0.30, 1.12, 2.65, school)
            west_sign("address_sign", group, -2.286, -0.65, 2.30, 0.96, 0.49, address)
        booth_y = side * 4.3
        ground_low, ground_high = terrain.extrema(3.3, yy + booth_y, 7.6, 7.0)
        low = ground_low - base - 0.12
        base_top = max(0.88, ground_high - base + 0.08)
        booth(label + "_booth", group, 3.3, booth_y, low, base_top, stone, glass, metal, roof)
        canopy(label + "_canopy", group, 2.2, side * 3.5, roof, metal)

        # Estimated small pedestrian control posts remain outside both lanes.
        for i, lane in enumerate((side * 8.30, side * 9.75)):
            z = terrain.height(-0.9, lane) - base
            item = empty(f"{label}_pedestrian_post_{i}", group, (-0.9, lane - yy, z))
            box(f"{label}_post_body_{i}", item, (0, 0, 0.50), (1.45, 0.22, 1.0), metal, 0.02)
            box(f"{label}_post_cap_{i}", item, (-0.30, 0, 1.04), (0.52, 0.27, 0.08), roof, 0.012)
            box(f"{label}_post_reader_{i}", item, (-0.48, 0, 0.84), (0.10, 0.13, 0.56), glass, 0.01)

    island_base = terrain.height(5.5, 0)
    median = empty("median", site, (5.5, 0, island_base))
    low = terrain.minimum(5.5, 0, 6.8, 1.85) - island_base - 0.08
    box("median_curb", median, (0, 0, (0.28 + low) / 2), (6.8, 1.85, 0.28 - low), stone, 0.035)
    box("median_kiosk_base", median, (1.4, 0, 0.655), (1.65, 1.30, 0.75), stone, 0.02)
    box("median_kiosk_glass", median, (1.4, 0, 1.48), (1.40, 1.18, 0.90), glass, 0.012)
    box("median_kiosk_roof", median, (1.4, 0, 2.005), (1.80, 1.50, 0.15), roof, 0.018)
    for side in (-1, 1):
        box("median_kiosk_corner_" + str(side + 1), median, (1.4, side * 0.59, 1.48),
            (0.055, 0.055, 0.9), metal, 0.006)

    for side, label in ((-1, "south"), (1, "north")):
        barrier_base(label + "_barrier", median, -1.5, side * 0.77, metal, roof)
        # The south arm is a true pivot. Its local +X axis maps to +X Y-up;
        # positive rotation lowers a vertical +Z arm toward the south lane.
        if side == -1:
            arm = empty("barrier_arm_pivot", median, (-1.5, -0.77, 1.04), "visual,interactive")
        else:
            arm = empty("north_barrier_static", median, (-1.5, 0.77, 1.04))
        box(label + "_barrier_arm", arm, (0, 0, 1.40), (0.095, 0.095, 2.80), white, 0.008)
        for i, z in enumerate((0.40, 1.25, 2.10)):
            box(f"{label}_barrier_red_band_{i}", arm, (0, 0, z), (0.106, 0.106, 0.32), red, 0.006)

    bpy.context.view_layer.update()


def preview_context(collection):
    """Read the separately owned snapshot only after saving and exporting."""
    context_json = CONTEXT / "context.json"
    if not context_json.is_file():
        return False
    context = json.loads(context_json.read_text(encoding="utf-8"))
    for layer in context["layers"]:
        path = CONTEXT / layer["mesh"]
        data = path.read_bytes()
        if data[:4] != b"SZM1":
            raise ValueError("Invalid context mesh: " + str(path))
        count = struct.unpack_from("<I", data, 4)[0]
        if len(data) != 8 + count * 32 or count % 3:
            raise ValueError("Invalid context mesh size: " + str(path))
        rows = list(struct.iter_unpack("<8f", data[8:]))
        # The input is local Y-up. Blender's authoring basis is X east, Y
        # north, Z up; the original UVs remain attached to each triangle loop.
        vertices = [(v[0], -v[2], v[1]) for v in rows]
        faces = [(i, i + 1, i + 2) for i in range(0, count, 3)]
        mesh = bpy.data.meshes.new("context_" + layer["id"])
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        uv = mesh.uv_layers.new(name="UVMap")
        for polygon in mesh.polygons:
            for loop in polygon.loop_indices:
                uv.data[loop].uv = rows[mesh.loops[loop].vertex_index][6:8]
        obj = bpy.data.objects.new("context_" + layer["id"], mesh)
        collection.objects.link(obj)
        obj["roles"] = "context"
        obj["source_role"] = layer["role"]
        color = tuple(layer["color"]) + (1.0,)
        mat = material("context_" + layer["id"], color, roughness=0.95)
        mat.use_backface_culling = False
        if layer["texture"]:
            image = bpy.data.images.load(str(CONTEXT / layer["texture"]), check_existing=True)
            tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
            tex.image = image
            multiply = mat.node_tree.nodes.new("ShaderNodeMixRGB")
            multiply.blend_type = "MULTIPLY"
            multiply.inputs[0].default_value = 1.0
            multiply.inputs[2].default_value = color
            mat.node_tree.links.new(tex.outputs["Color"], multiply.inputs[1])
            mat.node_tree.links.new(multiply.outputs[0], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
        mesh.materials.append(mat)
    print("Loaded", len(context["layers"]), "read-only preview context layers after export")
    return True


def add_preview(output):
    """Context or neutral studio preview, added only after the asset export."""
    preview = bpy.data.collections.new("context_preview_only")
    bpy.context.scene.collection.children.link(preview)
    has_context = preview_context(preview)
    mat = material("preview_ground", (0.28, 0.31, 0.28, 1), roughness=1)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -3 if has_context else -0.18))
    ground = bpy.context.object
    ground.name = "preview_ground_not_exported"
    ground.data.materials.append(mat)
    for collection in list(ground.users_collection):
        collection.objects.unlink(ground)
    preview.objects.link(ground)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 850
    scene.render.resolution_percentage = 100
    scene.world.color = (0.26, 0.26, 0.26)
    scene.view_settings.view_transform = "AgX"
    light_data = bpy.data.lights.new("preview_sun", "SUN")
    light = bpy.data.objects.new("preview_sun", light_data)
    preview.objects.link(light)
    light.rotation_euler = (math.radians(30), math.radians(-25), math.radians(-30))
    light_data.energy = 2.2
    cam_data = bpy.data.cameras.new("preview_camera")
    cam = bpy.data.objects.new("preview_camera", cam_data)
    preview.objects.link(cam)
    scene.camera = cam
    views = {
        "hero": ((-40, 22, 13), (2, 0, 4.2), 28),
        "front": ((-42, 0, 8), (1, 0, 5), 28),
        "top": ((0, -0.01, 90), (0, 0, 0), 30),
        "inside": ((55, -30, 16), (1, 0, 4), 35),
        # Eye height uses the actual cropped terrain at X=-22, Z=-3, +1.7 m.
        "walk": ((-22, 3, -0.00727234 + 1.7), (0, 0, 3.6), 28),
    }
    for name, (eye, target, lens) in views.items():
        cam.location = eye
        cam.rotation_euler = (Vector(target) - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam_data.lens = lens
        scene.render.filepath = str(output / "previews" / ("blender-" + name + ".png"))
        bpy.ops.render.render(write_still=True)


def main():
    args = arguments()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    terrain_path = CONTEXT / "meshes" / "terrain.mesh"
    terrain = Terrain(terrain_path, allow_flat=args.allow_flat_preview)
    build_scene(terrain)
    bpy.context.preferences.filepaths.save_version = 0
    # Save and export before adding any preview context, camera or ground.
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "model.blend"))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO / "tools" / "end" / "blender"))
    from export_asset import export_asset
    export_asset(ASSET / "asset.json", output)
    # Record actual sampled reference and foundation heights, distinguishing
    # the project's approximated terrain from any real-world survey.
    footprints = [
        ("north_outer", 0.0, 14.8, 4.5, 2.3, 0.0, 13.0),
        ("north_inner", 1.2, 11.2, 4.5, 2.3, 0.0, 13.0),
        ("north_booth", 3.3, 17.3, 7.6, 7.0, 0.0, 13.0),
        ("south_outer", 0.0, -12.1, 4.5, 4.4, 0.0, -13.0),
        ("south_return", 2.1, -15.0, 4.5, 3.0, 0.0, -13.0),
        ("south_booth", 3.3, -17.3, 7.6, 7.0, 0.0, -13.0),
        ("median", 5.5, 0.0, 6.8, 1.85, 5.5, 0.0),
    ]
    records = []
    for name, x, y, width, depth, ref_x, ref_y in footprints:
        low, high = terrain.extrema(x, y, width, depth)
        extension = 0.08 if name == "median" else 0.12
        reference = terrain.height(ref_x, ref_y)
        record = {"id": name, "reference_local_height_m": reference,
                  "footprint_ground_min_m": low, "footprint_ground_max_m": high,
                  "foundation_bottom_m": low - extension,
                  "extrema_method": "triangles clipped to the complete X/Y footprint"}
        if name.endswith("_booth"):
            record["stone_base_top_m"] = reference + max(0.88, high - reference + 0.08)
            record["stone_clearance_above_highest_ground_m"] = record["stone_base_top_m"] - high
        records.append(record)
    (output / "checks").mkdir(exist_ok=True)
    (output / "checks" / "grounding.json").write_text(json.dumps({
        "terrain": "context/meshes/terrain.mesh" if terrain.available else None,
        "terrain_sha256": hashlib.sha256(terrain_path.read_bytes()).hexdigest() if terrain.available else None,
        "coordinate_system": "Blender local metres, Z up", "context_is_project_approximation": True,
        "flat_preview_only": not terrain.available, "footprints": records,
    }, indent=2) + "\n", encoding="utf-8")
    add_preview(output)


if __name__ == "__main__":
    main()
