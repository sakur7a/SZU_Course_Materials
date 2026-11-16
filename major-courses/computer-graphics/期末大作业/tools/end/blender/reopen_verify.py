"""Reopen a saved fixture and import its GLB, then compare real scene data.

blender -b model.blend --python-exit-code 1 --python reopen_verify.py --
    --package DIRECTORY --report report.json
"""
import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import struct
import sys

import bpy
from mathutils import Matrix, Vector

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_asset import BASIS, BASIS_INV, matrix_yup, vector_yup


TOLERANCE = 1e-4


def assert_close(actual, expected, label, tolerance=TOLERANCE):
    if len(actual) != len(expected) or any(abs(a - b) > tolerance for a, b in zip(actual, expected)):
        raise AssertionError(f"{label}: {actual} != {expected}")


def verify_scene(manifest, stage):
    bpy.context.view_layer.update()
    objects = bpy.context.scene.objects
    for node in manifest["nodes"]:
        if node["id"] not in objects:
            raise AssertionError(f"{stage}: missing node {node['id']}")
        obj = objects[node["id"]]
        parent = obj.parent.name if obj.parent else None
        if parent != node["parent"]:
            raise AssertionError(f"{stage}: wrong parent for {obj.name}: {parent}")
        assert_close(matrix_yup(obj.matrix_local), node["local_matrix"], f"{stage} local {obj.name}")
    for anchor in manifest["anchors"]:
        obj = objects[anchor["node"]]
        local = BASIS_INV @ Vector((*anchor["local"], 1))
        actual = vector_yup(obj.matrix_world @ local.to_3d())
        assert_close(actual, anchor["world"], f"{stage} anchor {anchor['id']}")

    # Transform an actual ancestor and check an actual descendant, then restore.
    probe = manifest["hierarchy_probe"]
    parent, descendant = objects[probe["parent"]], objects[probe["descendant"]]
    ancestor = descendant.parent
    while ancestor and ancestor != parent:
        ancestor = ancestor.parent
    if not ancestor:
        raise AssertionError(stage + ": probe nodes do not have an ancestor relationship")
    original_local = parent.matrix_basis.copy()
    before_parent, before_child = parent.matrix_world.copy(), descendant.matrix_world.copy()
    parent.matrix_basis = original_local @ Matrix.Rotation(math.radians(probe["rotation_deg"]), 4, "Z")
    bpy.context.view_layer.update()
    actual = descendant.matrix_world.copy()
    expected = before_parent @ Matrix.Rotation(math.radians(probe["rotation_deg"]), 4, "Z") @ before_parent.inverted() @ before_child
    assert_close(matrix_yup(actual), matrix_yup(expected), f"{stage} hierarchy rotation")
    distance = (actual.translation - before_child.translation).length
    if distance < 1e-5:
        raise AssertionError(stage + ": hierarchy probe did not move descendant")
    parent.matrix_basis = original_local
    bpy.context.view_layer.update()
    assert_close(matrix_yup(descendant.matrix_world), matrix_yup(before_child), f"{stage} hierarchy restore")

    low, high = [float("inf")] * 3, [float("-inf")] * 3
    triangles = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            mesh.calc_loop_triangles()
            triangles += len(mesh.loop_triangles)
            for vertex in mesh.vertices:
                for axis, value in enumerate(vector_yup(obj.matrix_world @ vertex.co)):
                    low[axis] = min(low[axis], value)
                    high[axis] = max(high[axis], value)
        finally:
            evaluated.to_mesh_clear()
    assert_close(low, manifest["bounds"]["min"], f"{stage} min bounds")
    assert_close(high, manifest["bounds"]["max"], f"{stage} max bounds")
    if triangles != manifest["statistics"]["triangles"]:
        raise AssertionError(f"{stage} triangle count: {triangles}")
    return {"nodes": len(manifest["nodes"]), "triangles": triangles, "bounds": {"min": low, "max": high},
            "hierarchy_descendant_motion_m": distance}


def verify_glb_structure(package):
    blob = (package / "model.glb").read_bytes()
    magic, version, total = struct.unpack_from("<4sII", blob)
    if magic != b"glTF" or version != 2 or total != len(blob):
        raise AssertionError("Invalid GLB header")
    size, kind = struct.unpack_from("<I4s", blob, 12)
    if kind != b"JSON":
        raise AssertionError("Missing GLB JSON chunk")
    gltf = json.loads(blob[20:20 + size].decode("utf-8"))
    if any("uri" in image or "bufferView" not in image for image in gltf.get("images", [])):
        raise AssertionError("GLB image is not embedded")
    if any("uri" in buffer for buffer in gltf.get("buffers", [])):
        raise AssertionError("GLB buffer is not embedded")
    masks = [mat for mat in gltf.get("materials", []) if mat.get("alphaMode") == "MASK"]
    requirements = json.loads((package / "manifest.json").read_text(encoding="utf-8")).get("validation_requirements", {"mask": True})
    if requirements.get("mask", False) and not any(mat.get("pbrMetallicRoughness", {}).get("baseColorTexture") for mat in masks):
        raise AssertionError("GLB lacks a textured MASK material")
    return {"embedded_images": len(gltf.get("images", [])), "mask_materials": len(masks)}


def compare_legacy_geometry(manifest, package):
    """Compare imported GLB vertices/UVs/normals to the actual SZM1 payloads."""
    per_node = defaultdict(list)
    for primitive in manifest["primitives"]:
        data = (package / primitive["mesh"]).read_bytes()
        if len(data) < 8 or data[:4] != b"SZM1":
            raise AssertionError("Invalid SZM1: " + primitive["mesh"])
        count = struct.unpack_from("<I", data, 4)[0]
        if count % 3 or len(data) != 8 + count * 32:
            raise AssertionError("Invalid SZM1 vertex count: " + primitive["mesh"])
        per_node[primitive["node"]].extend(struct.iter_unpack("<8f", data[8:]))

    max_normal_delta, compared = 0.0, 0
    cell, position_tol, normal_tol = 1e-4, 2e-5, 0.002
    offsets = list(product((-1, 0, 1), repeat=3))
    for name, expected in per_node.items():
        obj = bpy.context.scene.objects[name]
        mesh = obj.data
        mesh.calc_loop_triangles()
        uv = mesh.uv_layers.active
        if uv is None:
            raise AssertionError("GLB mesh has no UVs: " + name)
        buckets = defaultdict(list)
        actual_count = 0
        for triangle in mesh.loop_triangles:
            for index in triangle.loops:
                loop = mesh.loops[index]
                pos = vector_yup(mesh.vertices[loop.vertex_index].co)
                normal = vector_yup(mesh.corner_normals[index].vector)
                texcoord = uv.data[index].uv
                record = (*pos, *normal, float(texcoord.x), float(texcoord.y))
                buckets[tuple(math.floor(p / cell) for p in pos)].append(record)
                actual_count += 1
        if actual_count != len(expected):
            raise AssertionError("GLB/SZM1 vertex count differs: " + name)
        for reference in expected:
            key = tuple(math.floor(p / cell) for p in reference[:3])
            match = None
            for offset in offsets:
                neighbor = tuple(key[i] + offset[i] for i in range(3))
                for index, candidate in enumerate(buckets.get(neighbor, ())):
                    if any(abs(reference[i] - candidate[i]) > position_tol for i in (0, 1, 2, 6, 7)):
                        continue
                    normal_delta = math.dist(reference[3:6], candidate[3:6])
                    if normal_delta <= normal_tol and (match is None or normal_delta < match[0]):
                        match = (normal_delta, neighbor, index)
            if match is None:
                raise AssertionError("GLB/SZM1 vertex, UV or normal differs: " + name)
            max_normal_delta = max(max_normal_delta, match[0])
            buckets[match[1]].pop(match[2])
            compared += 1
    return {"vertices": compared, "position_uv_tolerance": position_tol,
            "normal_tolerance": normal_tol, "max_normal_delta": max_normal_delta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    package, report = args.package.resolve(), args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "failed", "blender_version": bpy.app.version_string}
    try:
        if Path(bpy.data.filepath).resolve() != package / "model.blend":
            raise AssertionError("The running Blender process did not open the package model.blend")
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        for relative, expected in manifest["files"].items():
            path = (package / relative).resolve()
            if not path.is_relative_to(package) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise AssertionError("Missing, external or changed file: " + relative)
        result["blend"] = verify_scene(manifest, "blend")
        result["glb_structure"] = verify_glb_structure(package)
        for mat in manifest["materials"]:
            if mat["alpha_mode"] == "MASK":
                data = (package / mat["legacy_texture"]).read_bytes()
                width, height = struct.unpack_from("<II", data)
                alpha = data[11::4]
                if len(data) != 8 + width * height * 4 or 0 not in alpha or 255 not in alpha:
                    raise AssertionError("MASK texture does not contain transparent and opaque texels")
        # Discard the reopened objects before importing GLB so names and hierarchy
        # are checked on the actual independently decoded asset.
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.import_scene.gltf(filepath=str(package / "model.glb"))
        result["glb"] = verify_scene(manifest, "glb")
        result["glb_vs_legacy"] = compare_legacy_geometry(manifest, package)
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = str(exc)
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Reopen and GLB verification passed:", report)


if __name__ == "__main__":
    main()
