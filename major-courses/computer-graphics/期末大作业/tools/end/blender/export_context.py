"""Export a read-only SZM1 context snapshot as its own self-contained GLB.

Run Blender in a separate background process, with --context, --output and
--report after Blender's ``--`` separator. This never exports the gate model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import bpy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Context file is missing or outside its directory: " + str(relative))
    return path


def read_vertices(path, expected_hash):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("Context mesh hash changed: " + str(path))
    if len(data) < 8 or data[:4] != b"SZM1":
        raise ValueError("Invalid SZM1 header: " + str(path))
    count = struct.unpack_from("<I", data, 4)[0]
    if not count or count % 3 or len(data) != 8 + count * 32:
        raise ValueError("Invalid SZM1 mesh size: " + str(path))
    vertices = list(struct.iter_unpack("<8f", memoryview(data)[8:]))
    if not all(math.isfinite(x) for vertex in vertices for x in vertex):
        raise ValueError("Non-finite context mesh value: " + str(path))
    if any(sum(x*x for x in vertex[3:6]) < 1e-12 for vertex in vertices):
        raise ValueError("Zero-length context normal: " + str(path))
    return vertices


def material_for(layer, texture_path):
    material = bpy.data.materials.new("ctx_material_" + layer["id"])
    material.use_nodes = True
    material.use_backface_culling = False  # The legacy campus does not enable culling.
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Metallic"].default_value = 0
    shader.inputs["Roughness"].default_value = 1
    color = (*layer["color"], 1.0)
    if texture_path is None:
        shader.inputs["Base Color"].default_value = color
    else:
        image = bpy.data.images.load(str(texture_path), check_existing=True)
        image.colorspace_settings.name = "sRGB"
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.image = image
        image_node.extension = "REPEAT"
        # The glTF exporter recognises a color constant multiplied by an image
        # and writes baseColorFactor + baseColorTexture without baking pixels.
        multiply = nodes.new("ShaderNodeMix")
        multiply.data_type = "RGBA"
        multiply.blend_type = "MULTIPLY"
        multiply.inputs[0].default_value = 1.0
        multiply.inputs[6].default_value = color
        links.new(image_node.outputs["Color"], multiply.inputs[7])
        links.new(multiply.outputs[2], shader.inputs["Base Color"])
    return material


def create_layer(layer, records, material, collection):
    # A proper rotation converts local Y-up to Blender Z-up. Blender's glTF
    # exporter performs the inverse once; context coordinates are never placed
    # at their campus world origin inside this companion GLB.
    coords = [(v[0], -v[2], v[1]) for v in records]
    normals = [(v[3], -v[5], v[4]) for v in records]
    faces = [(i, i + 1, i + 2) for i in range(0, len(records), 3)]
    name = "ctx_" + layer["id"]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(coords, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    for loop in mesh.loops:
        vertex = records[loop.vertex_index]
        uv.data[loop.index].uv = (vertex[6], vertex[7])
    # Blender/glTF normalise custom normals; the input vectors may be slightly
    # non-unit after clipping, but their directions are preserved.
    mesh.normals_split_custom_set(normals)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    mesh.materials.append(material)
    obj["context_role"] = layer["role"]
    obj["writable"] = False
    return obj


def export(context_path, output, report):
    context_path, output, report = context_path.resolve(), output.resolve(), report.resolve()
    context_dir = context_path.parent
    if output == report or output.is_relative_to(context_dir) or report.is_relative_to(context_dir):
        raise ValueError("Output and report must be separate and outside the read-only context directory")
    campus_assets = Path(__file__).resolve().parents[3] / "main/end/code/assets"
    if (output.name != "context.glb" or report.name != "context.json" or
            report.parent.name != "checks" or report.parent.parent != output.parent or
            "legacy" in output.parts or output.is_relative_to(campus_assets) or report.is_relative_to(campus_assets)):
        raise ValueError("Expected <run>/context.glb and <run>/checks/context.json outside campus/legacy assets")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if context.get("coordinate_system") != "RH_Y_UP" or context.get("writable") is not False:
        raise ValueError("Expected a read-only Y-up context")
    textures = {entry["path"]: entry for entry in context["textures"].values()}
    # This process owns a separate in-memory scene, and selection further
    # restricts the export to the newly created context nodes.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    collection = bpy.data.collections.new("ReadOnlyContext")
    bpy.context.scene.collection.children.link(collection)

    layer_results, objects = [], []
    overall_min, overall_max = [math.inf]*3, [-math.inf]*3
    ids = set()
    for layer in context["layers"]:
        layer_id = layer["id"]
        if not layer_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in layer_id) or layer_id in ids:
            raise ValueError("Invalid or duplicate context layer ID: " + str(layer_id))
        ids.add(layer_id)
        if layer.get("writable") is not False:
            raise ValueError("Context layer must be read-only: " + layer_id)
        mesh_path = input_path(context_dir, layer["mesh"])
        records = read_vertices(mesh_path, layer["mesh_sha256"])
        if len(records) // 3 != layer["triangles"]:
            raise ValueError("Context triangle count changed: " + layer_id)
        texture = layer.get("texture")
        texture_path = input_path(context_dir, texture) if texture else None
        if texture_path and digest(texture_path) != textures[texture]["sha256"]:
            raise ValueError("Context texture hash changed: " + str(texture))
        obj = create_layer(layer, records, material_for(layer, texture_path), collection)
        objects.append(obj)
        low = [min(v[i] for v in records) for i in range(3)]
        high = [max(v[i] for v in records) for i in range(3)]
        overall_min = [min(a, b) for a, b in zip(overall_min, low)]
        overall_max = [max(a, b) for a, b in zip(overall_max, high)]
        layer_results.append({"id": layer_id, "node": obj.name, "role": layer["role"],
                              "triangles": len(records)//3, "material": obj.data.materials[0].name,
                              "color": layer["color"], "texture": texture,
                              "bounds_yup": {"min": low, "max": high}})
    if not objects:
        raise ValueError("Context has no layers")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.export_scene.gltf(filepath=str(output), export_format="GLB", export_yup=True,
                                       use_selection=True, export_cameras=False, export_lights=False)
    if "FINISHED" not in result or not output.is_file():
        raise RuntimeError("Blender failed to export the context GLB")
    result = {"status": "exported", "read_only": True, "coordinate_system": "RH_Y_UP",
              "coordinates": "local to origin_yup; add origin once for campus world coordinates",
              "origin_yup": context["origin_yup"],
              "bounds_yup": {"min": overall_min, "max": overall_max},
              "statistics": {"nodes": len(objects), "meshes": len(objects),
                             "materials": len(objects), "triangles": sum(x["triangles"] for x in layer_results)},
              "layers": layer_results, "source_context_sha256": digest(context_path),
              "glb_sha256": digest(output), "blender_version": bpy.app.version_string,
              "normal_note": "Blender/glTF normalise and quantise source normal vectors; directions may differ slightly.",
              "material_note": "Base-color texture and multiplier are retained; PBR lighting is an approximation of the legacy renderer."}
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Exported read-only context:", output, result["statistics"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    export(args.context, args.output, args.report)


if __name__ == "__main__":
    main()
