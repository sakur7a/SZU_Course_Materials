"""Export a saved Blender scene as GLB and M0's local-mesh legacy format.

Supported materials are Principled base colors with at most one color image;
MASK uses an image alpha socket through ROUND. Other PBR channels are recorded
as approximations in the manifest. Run inside Blender or import export_asset.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import bpy
from mathutils import Matrix, Vector


# Blender (X right, Y forward, Z up) -> RH glTF / course Y-up.
BASIS = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
BASIS_INV = BASIS.inverted()


def vector_yup(vec):
    return [float(vec[0]), float(vec[2]), -float(vec[1])]


def matrix_yup(matrix):
    converted = BASIS @ matrix @ BASIS_INV
    return [float(converted[row][column]) for column in range(4) for row in range(4)]


def fmt(value):
    return format(float(value), ".9g")


def checked_name(name):
    if not name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in name):
        raise ValueError("IDs must be stable ASCII letters, digits, '_' or '-': " + name)
    return name


def texture_rgba(image, path):
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions: " + image.name)
    pixels = array("f", [0.0]) * (width * height * 4)
    image.pixels.foreach_get(pixels)
    if image.colorspace_settings.name != "sRGB" or image.is_float:
        raise ValueError("Legacy color images must be byte-encoded sRGB: " + image.name)

    def byte(value):
        return round(min(1.0, max(0.0, value)) * 255)

    raw = bytearray(len(pixels))
    # For byte-backed sRGB images, Image.pixels exposes encoded pixel values,
    # not linearized shader values. Rows already start at the OpenGL bottom.
    for i in range(0, len(pixels), 4):
        raw[i:i + 4] = bytes(byte(pixels[i + channel]) for channel in range(4))
    path.write_bytes(struct.pack("<II", width, height) + raw)
    return [width, height]


def export_material(mat, output):
    checked_name(mat.name)
    shader = next((node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None) if mat.use_nodes else None
    if shader is None:
        raise ValueError("A Principled material is required: " + mat.name)
    image_nodes = [n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.image is not None]
    if len(image_nodes) > 1:
        raise ValueError("Only one base color image per material is supported: " + mat.name)
    alpha_mode = str(mat.get("alpha_mode", "OPAQUE"))
    if alpha_mode not in ("OPAQUE", "MASK"):
        raise ValueError("Unsupported alpha mode in " + mat.name)
    image = image_nodes[0].image if image_nodes else None
    relative = "legacy/textures/" + checked_name(mat.name) + ".rgba" if image else None
    texture_size = texture_rgba(image, output / relative) if image else None
    # An image directly linked to Base Color is the complete glTF base color.
    color = [1.0, 1.0, 1.0, 1.0] if image else list(shader.inputs["Base Color"].default_value)
    return {"id": mat.name, "base_color": [float(x) for x in color], "alpha_mode": alpha_mode,
            "alpha_cutoff": float(mat.get("alpha_cutoff", 0.5)), "double_sided": not mat.use_backface_culling,
            "legacy_texture": relative, "texture_size": texture_size,
            "metallic": float(shader.inputs["Metallic"].default_value),
            "roughness": float(shader.inputs["Roughness"].default_value)}


def write_meshes(obj, depsgraph, output):
    # Export evaluated local vertices rather than baking the node world matrix;
    # the runtime traverses local matrices once and uses inverse-transpose normals.
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()
        uv_layer = mesh.uv_layers.active
        if uv_layer is None:
            raise ValueError("A UV map is required: " + obj.name)
        grouped = {}
        for tri in mesh.loop_triangles:
            if tri.material_index >= len(obj.material_slots) or obj.material_slots[tri.material_index].material is None:
                raise ValueError("A material is required: " + obj.name)
            material = obj.material_slots[tri.material_index].material
            group = grouped.setdefault(material.name, bytearray())
            for index in tri.loops:
                loop = mesh.loops[index]
                pos = vector_yup(mesh.vertices[loop.vertex_index].co)
                normal = vector_yup(mesh.corner_normals[index].vector)
                uv = uv_layer.data[index].uv
                values = (*pos, *normal, float(uv.x), float(uv.y))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Non-finite vertex data: " + obj.name)
                group.extend(struct.pack("<8f", *values))
        primitives = []
        for material, payload in sorted(grouped.items()):
            primitive_id = checked_name(obj.name) + "_" + checked_name(material)
            relative = "legacy/meshes/" + primitive_id + ".mesh"
            (output / relative).write_bytes(b"SZM1" + struct.pack("<I", len(payload) // 32) + payload)
            primitives.append({"id": primitive_id, "node": obj.name, "material": material, "mesh": relative})
        return primitives
    finally:
        evaluated.to_mesh_clear()


def write_tsv(path, rows):
    path.write_text("\n".join("\t".join(str(value) for value in row) for row in rows) + "\n", encoding="utf-8", newline="\n")


def export_asset(asset_path, output):
    asset = json.loads(Path(asset_path).read_text(encoding="utf-8"))
    output = Path(output).resolve()
    for folder in ("legacy/meshes", "legacy/textures"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    bpy.context.view_layer.update()
    objects = sorted((o for o in bpy.context.scene.objects if o.type in {"MESH", "EMPTY"}), key=lambda o: o.name)
    names = {checked_name(o.name) for o in objects}
    for obj in objects:
        if obj.parent and obj.parent.name not in names:
            raise ValueError("Exported node has a missing parent: " + obj.name)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    nodes = [{"id": obj.name, "parent": obj.parent.name if obj.parent else None,
              "local_matrix": matrix_yup(obj.matrix_local),
              "roles": [x for x in str(obj.get("roles", "visual")).split(",") if x]} for obj in objects]
    materials = sorted({slot.material for obj in objects if obj.type == "MESH" for slot in obj.material_slots
                        if slot.material}, key=lambda mat: mat.name)
    materials = [export_material(mat, output) for mat in materials]
    primitives = []
    bounds_min, bounds_max = [float("inf")] * 3, [float("-inf")] * 3
    triangle_count = 0
    for obj in objects:
        if obj.type != "MESH":
            continue
        primitives.extend(write_meshes(obj, depsgraph, output))
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            mesh.calc_loop_triangles()
            triangle_count += len(mesh.loop_triangles)
            for vertex in mesh.vertices:
                coords = vector_yup(obj.matrix_world @ vertex.co)
                for axis in range(3):
                    bounds_min[axis] = min(bounds_min[axis], coords[axis])
                    bounds_max[axis] = max(bounds_max[axis], coords[axis])
        finally:
            evaluated.to_mesh_clear()
    if not triangle_count:
        raise ValueError("Scene has no mesh triangles")
    anchors = []
    for anchor in asset["anchors"]:
        obj = bpy.data.objects[anchor["node"]]
        local = Vector(anchor["local_blender"])
        anchors.append({"id": anchor["id"], "node": obj.name, "local": vector_yup(local),
                        "world": vector_yup(obj.matrix_world @ local)})

    # The standard exporter performs Y-up conversion; never rotate this scene first.
    bpy.ops.export_scene.gltf(filepath=str(output / "model.glb"), export_format="GLB", export_yup=True,
                              export_apply=True, export_cameras=False, export_lights=False)
    write_tsv(output / "legacy/nodes.tsv", [["N", n["id"], n["parent"] or "-", ",".join(n["roles"]),
              *map(fmt, n["local_matrix"])] for n in nodes])
    write_tsv(output / "legacy/materials.tsv", [["T", m["id"], m["legacy_texture"] or "-", m["alpha_mode"],
              fmt(m["alpha_cutoff"]), int(m["double_sided"]), *map(fmt, m["base_color"])] for m in materials])
    write_tsv(output / "legacy/primitives.tsv", [["P", p["id"], p["node"], p["material"], p["mesh"]] for p in primitives])
    write_tsv(output / "legacy/anchors.tsv", [["A", a["id"], a["node"], *map(fmt, a["local"]),
              *map(fmt, a["world"])] for a in anchors])
    interaction, probe, views = asset["interaction"], asset["hierarchy_probe"], asset["views"]
    controls = [["I", interaction["node"], *map(fmt, interaction["axis"]), *map(fmt, interaction["range_deg"])],
                ["H", probe["parent"], probe["descendant"], fmt(probe["rotation_deg"])]]
    controls += [["V", name, *map(fmt, view["eye"]), *map(fmt, view["target"])] for name, view in views.items()]
    write_tsv(output / "legacy/controls.tsv", controls)
    manifest = {"schema_version": 1, "asset_id": asset["asset_id"], "asset_version": asset["asset_version"],
                "name": asset["name"], "synthetic": asset["synthetic"], "units": "m", "coordinate_system": "RH_Y_UP",
                "model": "model.glb", "bounds": {"min": bounds_min, "max": bounds_max}, "nodes": nodes,
                "materials": materials, "primitives": primitives, "anchors": anchors, "interaction": interaction,
                "hierarchy_probe": probe, "views": views,
                "validation_requirements": asset.get("validation_requirements", {"mask": True, "min_hierarchy_depth": 4}),
                "statistics": {"triangles": triangle_count, "meshes": sum(obj.type == "MESH" for obj in objects)},
                "conversion_report": {"supported": ["base color", "sRGB color texture", "alpha MASK", "double sided"],
                  "approximated": ["metallic and roughness are reported but not represented by the legacy material"],
                  "unsupported": ["normal maps and other PBR channels"]}}
    if "placement" in asset:
        manifest["placement"] = asset["placement"]
    if "evidence" in asset:
        manifest["evidence"] = asset["evidence"]
    # Hash actual deliverables, excluding the self-referential manifest and reports.
    paths = [output / "model.blend", output / "model.glb", *sorted((output / "legacy").rglob("*"))]
    manifest["files"] = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in paths if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Exported", len(nodes), "nodes,", triangle_count, "triangles to", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    export_asset(args.asset, args.output)


if __name__ == "__main__":
    main()
