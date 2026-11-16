"""Generate the synthetic M0 pavilion in Blender 4.5, without campus inputs.

Run with Blender, not system Python: --python build.py -- --output DIRECTORY.
The saved scene is the common source for the GLB and evaluated-mesh exports.
"""
import argparse
import math
from pathlib import Path
import struct
import sys
import zlib

import bpy


def args_after_separator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def png_rgba(path, width, height, rows):
    """Write sRGB RGBA bytes with the first row at the top, using stdlib only."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    scanlines = b"".join(b"\0" + row for row in rows)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    data += chunk(b"sRGB", b"\0") + chunk(b"IDAT", zlib.compress(scanlines, 9)) + chunk(b"IEND", b"")
    path.write_bytes(data)


def texture_images(output):
    folder = output / "legacy" / "textures"
    folder.mkdir(parents=True, exist_ok=True)
    n = 512
    stone_rows, screen_rows = [], []
    for y in range(n):
        stone, screen = bytearray(), bytearray()
        for x in range(n):
            # Restrained tile joints and deterministic speckle, stored in sRGB.
            grout = x % 128 < 3 or y % 128 < 3
            noise = ((x * 37 + y * 61 + (x * y) % 31) % 9) - 4
            base = (169, 157, 136) if grout else (215, 207, 188)
            stone.extend((*(max(0, min(255, c + noise)) for c in base), 255))
            # The diagonal lattice is genuinely transparent between opaque lines.
            line = ((x + y) % 56 < 5) or ((x - y) % 56 < 5)
            screen.extend((50, 96, 91, 255 if line else 0))
        stone_rows.append(bytes(stone))
        screen_rows.append(bytes(screen))
    images = {}
    for name, rows in (("limestone", stone_rows), ("lattice", screen_rows)):
        path = folder / (name + ".png")
        png_rgba(path, n, n, rows)
        image = bpy.data.images.load(str(path), check_existing=False)
        image.name = name
        image.colorspace_settings.name = "sRGB"
        # Pack the exact PNG into the blend while retaining a package-relative path.
        image.pack()
        image.filepath = "//legacy/textures/" + name + ".png"
        images[name] = image
    return images


def material(name, color, image=None, mask=False, metallic=0.0, roughness=0.75):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = not mask
    mat.diffuse_color = color
    mat["alpha_mode"] = "MASK" if mask else "OPAQUE"
    mat["alpha_cutoff"] = 0.5
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    if image:
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.interpolation = "Linear"
        mat.node_tree.links.new(tex.outputs["Color"], shader.inputs["Base Color"])
        if mask:
            # Blender 4.5's glTF exporter recognizes ROUND as alphaMode MASK.
            clip = mat.node_tree.nodes.new("ShaderNodeMath")
            clip.operation = "ROUND"
            mat.node_tree.links.new(tex.outputs["Alpha"], clip.inputs[0])
            mat.node_tree.links.new(clip.outputs[0], shader.inputs["Alpha"])
            mat.surface_render_method = "DITHERED"
    return mat


def empty(name, parent=None, location=(0, 0, 0), angle=0, roles="visual"):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    obj.location = location
    obj.rotation_euler[2] = math.radians(angle)
    obj["roles"] = roles
    return obj


def box(name, parent, location, dimensions, mat, bevel=0.0, roles="visual"):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.name = name
    obj.scale = dimensions
    # Apply dimensions to local vertices; node transforms remain meaningful.
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.parent = parent
    obj.location = location
    obj.data.materials.append(mat)
    obj["roles"] = roles
    if bevel:
        modifier = obj.modifiers.new("soft_edges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 2
        obj.modifiers.new("weighted_corners", "WEIGHTED_NORMAL")
    return obj


def lattice_plane(parent, mat):
    # Local X/Z plane, front-facing toward -Y; UV origin is bottom-left.
    mesh = bpy.data.meshes.new("lattice_screen_mesh")
    mesh.from_pydata([(-2.1, 0, 0), (2.1, 0, 0), (2.1, 0, 1.65), (-2.1, 0, 1.65)], [], [(0, 1, 2, 3)])
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    for loop, xy in zip(mesh.polygons[0].loop_indices, ((0, 0), (1, 0), (1, 1), (0, 1))):
        uv.data[loop].uv = xy
    obj = bpy.data.objects.new("lattice_screen", mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    obj.location = (0, 1.52, 0.38)
    obj["roles"] = "visual"
    obj.data.materials.append(mat)


def build_scene(output):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    textures = texture_images(output)
    limestone = material("limestone", (1, 1, 1, 1), textures["limestone"])
    lattice = material("lattice_mask", (1, 1, 1, 1), textures["lattice"], mask=True)
    pale = material("warm_plaster", (0.72, 0.67, 0.56, 1), roughness=0.82)
    dark = material("deep_teal", (0.045, 0.16, 0.145, 1), metallic=0.12, roughness=0.42)
    brass = material("brass", (0.55, 0.31, 0.10, 1), metallic=0.5, roughness=0.4)
    ground = material("ground", (0.25, 0.31, 0.24, 1), roughness=0.95)

    # All dimensions are metres. Blender X/Y are horizontal, Z is up.
    root = empty("root")
    box("ground_plane", root, (0, 0, -0.08), (9.4, 7.4, 0.16), ground, 0.02, "visual,walkable")
    site = empty("site", root, (0.18, 0.08, 0.0))
    box("platform_base", site, (0, 0, 0.10), (5.9, 4.5, 0.20), pale, 0.04)
    box("platform_paving", site, (0, 0, 0.225), (5.72, 4.32, 0.05), limestone, 0.008, "visual,walkable")
    # Four real transforming ancestors: site -> building -> frame -> canopy.
    building = empty("building", site, (0.10, 0.05, 0.25), angle=5)
    frame = empty("frame", building, (0.04, 0.03, 0.02), angle=-3)
    for side, x in (("left", -2.40), ("right", 2.40)):
        for end, y in (("front", -1.55), ("rear", 1.55)):
            box(f"post_{side}_{end}", frame, (x, y, 1.31), (0.20, 0.20, 2.62), dark, 0.025)
            box(f"foot_{side}_{end}", frame, (x, y, 0.08), (0.31, 0.31, 0.16), brass, 0.018)
    for y, label in ((-1.55, "front"), (1.55, "rear")):
        box("beam_" + label, frame, (0, y, 2.52), (5.0, 0.19, 0.20), dark, 0.02)
        box("beam_trim_" + label, frame, (0, y - 0.11, 2.49), (4.85, 0.018, 0.028), brass)
    for x, label in ((-2.40, "left"), (2.40, "right")):
        box("beam_" + label, frame, (x, 0, 2.52), (0.19, 3.2, 0.20), dark, 0.02)
    canopy = empty("canopy", frame, (0, 0, 2.65), angle=2)
    box("canopy_roof", canopy, (0, 0, 0.08), (5.85, 4.15, 0.16), dark, 0.04)
    box("canopy_cap", canopy, (0, 0, 0.175), (5.55, 3.86, 0.05), pale, 0.015)
    box("canopy_front_trim", canopy, (0, -2.08, 0.08), (5.65, 0.025, 0.035), brass)
    lattice_plane(frame, lattice)
    for x, label in ((-2.13, "left"), (2.13, "right")):
        box("screen_border_" + label, frame, (x, 1.51, 1.20), (0.06, 0.06, 1.75), dark, 0.006)
    for z, label in ((0.35, "lower"), (2.06, "upper")):
        box("screen_border_" + label, frame, (0, 1.51, z), (4.32, 0.06, 0.06), dark, 0.006)

    # The hinge object is at the left edge, so runtime Y-axis rotation swings
    # every visible slat around the same pivot after the Y-up conversion.
    for side, x in (("left", -0.99), ("right", 0.99)):
        box("gate_post_" + side, frame, (x, -1.55, 0.79), (0.12, 0.12, 1.58), dark, 0.012)
        box("gate_post_cap_" + side, frame, (x, -1.55, 1.59), (0.15, 0.15, 0.035), brass, 0.007)
        rail_x = -1.70 if x < 0 else 1.70
        for z, label in ((0.25, "lower"), (1.40, "upper")):
            box(f"fence_{side}_{label}", frame, (rail_x, -1.55, z), (1.35, 0.07, 0.07), dark, 0.006)
        for i in range(6):
            post_x = (-2.28 + i * 0.21) if x < 0 else (1.22 + i * 0.21)
            box(f"fence_{side}_slat_{i:02d}", frame, (post_x, -1.55, 0.82), (0.05, 0.06, 1.16), dark, 0.006)
    hinge = empty("gate_hinge", frame, (-0.92, -1.56, 0.06), roles="visual,interactive")
    for z, label in ((0.19, "bottom"), (1.42, "top")):
        box("gate_rail_" + label, hinge, (0.92, 0, z), (1.84, 0.08, 0.08), dark, 0.008)
    for i in range(10):
        box(f"gate_slat_{i:02d}", hinge, (0.08 + i * 0.185, 0, 0.805), (0.065, 0.07, 1.28), dark, 0.008)
    box("gate_handle", hinge, (1.73, -0.06, 0.89), (0.055, 0.055, 0.23), brass, 0.012)
    bpy.context.view_layer.update()


def main():
    output = args_after_separator().output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build_scene(output)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "model.blend"))
    repository = Path(__file__).resolve().parents[6]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repository / "tools" / "end" / "blender"))
    from export_asset import export_asset
    export_asset(Path(__file__).resolve().parents[1] / "asset.json", output)


if __name__ == "__main__":
    main()
