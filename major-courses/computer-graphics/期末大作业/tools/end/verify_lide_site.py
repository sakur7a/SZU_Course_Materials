"""Independently inspect the exported Lide gate against its terrain snapshot.

This is a bounded, sampled static-geometry check, not a physics engine or a
claim about real-world access. The model builder's grounding report is not read.
"""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import struct
import sys

from asset_checks import check_legacy_tables, inside, multiply, node_matrix, point, read_glb, sha256, worlds


WALK_LAYERS = ("terrain", "road-surface", "network-sidewalk", "network-paths", "lide-walk")
BASES = ("north_outer_wall", "north_inner_wall", "south_outer_wall", "south_return_wall",
         "north_booth_stone_base", "south_booth_stone_base", "median_curb", "median_kiosk_base")
GLASS = ("north_booth_glazing", "south_booth_glazing", "median_kiosk_glass")
TOL = 1e-4


def read_mesh(path):
    data = path.read_bytes()
    if len(data) < 8 or data[:4] != b"SZM1":
        raise ValueError("Invalid SZM1 header: " + str(path))
    count = struct.unpack_from("<I", data, 4)[0]
    if not count or count % 3 or len(data) != 8 + 32 * count:
        raise ValueError("Invalid SZM1 count: " + str(path))
    values = list(struct.iter_unpack("<8f", memoryview(data)[8:]))
    if not all(math.isfinite(v) for row in values for v in row):
        raise ValueError("Non-finite mesh value: " + str(path))
    return values


def as_triangles(values):
    return [tuple(values[i:i+3]) for i in range(0, len(values), 3)]


def bounds(vertices):
    return {"min": [min(v[i] for v in vertices) for i in range(3)],
            "max": [max(v[i] for v in vertices) for i in range(3)]}


def barycentric_y(triangle, x, z):
    a, b, c = triangle
    det = (b[2]-c[2])*(a[0]-c[0]) + (c[0]-b[0])*(a[2]-c[2])
    if abs(det) < 1e-10:
        return None
    u = ((b[2]-c[2])*(x-c[0]) + (c[0]-b[0])*(z-c[2])) / det
    v = ((c[2]-a[2])*(x-c[0]) + (a[0]-c[0])*(z-c[2])) / det
    w = 1-u-v
    if min(u, v, w) < -1e-7:
        return None
    return u*a[1] + v*b[1] + w*c[1]


class Surface:
    """Small spatial index for exact triangle-height queries (no extrapolation)."""
    def __init__(self, triangles):
        self.triangles, self.cells = triangles, {}
        for triangle in triangles:
            xs, zs = [p[0] for p in triangle], [p[2] for p in triangle]
            for i in range(math.floor(min(xs)/5), math.floor(max(xs)/5)+1):
                for j in range(math.floor(min(zs)/5), math.floor(max(zs)/5)+1):
                    self.cells.setdefault((i, j), []).append(triangle)

    def heights(self, x, z):
        hits = []
        for tri in self.cells.get((math.floor(x/5), math.floor(z/5)), ()):
            y = barycentric_y(tri, x, z)
            if y is not None:
                hits.append(y)
        return hits


def load_context(path):
    context = json.loads(path.read_text(encoding="utf-8"))
    if context.get("coordinate_system") != "RH_Y_UP" or context.get("writable") is not False:
        raise ValueError("Expected a read-only Y-up context")
    origin = context["origin_yup"]
    if len(origin) != 3 or not all(math.isfinite(x) for x in origin):
        raise ValueError("Invalid context origin")
    layers = {layer["id"]: layer for layer in context["layers"]}
    surfaces = {}
    for name in WALK_LAYERS:
        layer = layers[name]
        path_mesh = inside(path.parent, layer["mesh"])
        if sha256(path_mesh) != layer["mesh_sha256"]:
            raise ValueError("Context mesh hash mismatch: " + name)
        verts = [[v[i] + origin[i] for i in range(3)] for v in read_mesh(path_mesh)]
        surfaces[name] = Surface(as_triangles(verts))
    return context, surfaces


def placement_for(package, manifest, origin, issues):
    declared, table = manifest.get("placement"), package / "placement.tsv"
    expected = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., *origin, 1.]
    matrix = node_matrix(declared) if declared else None
    if matrix is not None and not all(math.isfinite(x) for x in matrix):
        raise ValueError("Non-finite manifest placement")
    table_matrix = None
    if table.is_file():
        lines = [line.split("\t") for line in table.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != 1 or len(lines[0]) != 17 or lines[0][0] != "P":
            raise ValueError("Invalid placement.tsv")
        table_matrix = [float(x) for x in lines[0][1:]]
        if not all(math.isfinite(x) for x in table_matrix):
            raise ValueError("Non-finite placement matrix")
        if "placement.tsv" not in manifest["files"] or sha256(table) != manifest["files"]["placement.tsv"]:
            raise ValueError("placement.tsv is not fingerprinted or its hash changed")
    if matrix and table_matrix and max(abs(a-b) for a, b in zip(matrix, table_matrix)) > TOL:
        issues.append("Manifest placement and placement.tsv disagree")
    actual = matrix or table_matrix or expected
    mismatch = max(abs(a-b) for a, b in zip(actual, expected))
    if mismatch > TOL:
        issues.append("Placement differs from the context origin and identity rotation/scale")
    complete = matrix is not None and table_matrix is not None
    return actual, {"status": "verified" if complete and mismatch <= TOL and not issues else
                              "assumed" if not matrix and not table_matrix else "incomplete_or_mismatched",
                    "source": "manifest and placement.tsv" if complete else
                              "manifest only" if matrix else "placement.tsv only" if table_matrix else "assumed context origin",
                    "matrix_column_major": actual, "context_origin_yup": origin,
                    "expected_matrix_column_major": expected, "maximum_matrix_difference": mismatch,
                    "note": "An assumed placement is for analysis only; no actual campus placement is thereby verified."}


def inspect_model(package, manifest, placement):
    if manifest.get("asset_id") != "lide-gate" or manifest.get("coordinate_system") != "RH_Y_UP":
        raise ValueError("Expected a Lide Gate Y-up package")
    needed = {manifest["model"], "legacy/nodes.tsv", "legacy/primitives.tsv", "legacy/anchors.tsv",
              "legacy/materials.tsv", "legacy/controls.tsv"}
    needed.update(p["mesh"] for p in manifest["primitives"])
    for name in needed:
        path = inside(package, name)
        if name not in manifest["files"] or not path.is_file() or sha256(path) != manifest["files"][name]:
            raise ValueError("Package mesh/table/model hash mismatch: " + name)
    check_legacy_tables(package, manifest)
    nodes = {item["id"]: item for item in manifest["nodes"]}
    if len(nodes) != len(manifest["nodes"]) or any(name.startswith("ctx_") for name in nodes):
        raise ValueError("Duplicate node or context node in model manifest")
    local_worlds = worlds(nodes)
    doc, _ = read_glb(inside(package, manifest["model"]))
    glb_nodes = doc.get("nodes", [])
    names = [item.get("name") for item in glb_nodes]
    if len(set(names)) != len(names) or set(names) != set(nodes) or any(str(n).startswith("ctx_") for n in names):
        raise ValueError("Main GLB includes context nodes or differs from model node identities")
    glb_tree = {str(i): {"parent": None, "local_matrix": node_matrix(n)} for i, n in enumerate(glb_nodes)}
    for i, node in enumerate(glb_nodes):
        for child in node.get("children", []):
            if glb_tree[str(child)]["parent"] is not None:
                raise ValueError("GLB node has multiple parents")
            glb_tree[str(child)]["parent"] = str(i)
    glb_worlds = worlds(glb_tree)
    for i, name in enumerate(names):
        if max(abs(a-b) for a, b in zip(local_worlds[name], glb_worlds[str(i)])) > TOL:
            raise ValueError("GLB/legacy model transformation differs: " + name)

    vertices_by_node, triangles = {}, 0
    for primitive in manifest["primitives"]:
        matrix = multiply(placement, local_worlds[primitive["node"]])
        records = read_mesh(inside(package, primitive["mesh"]))
        vertices_by_node.setdefault(primitive["node"], []).extend(point(matrix, v[:3]) for v in records)
        triangles += len(records)//3
    if triangles != manifest["statistics"]["triangles"]:
        raise ValueError("Model triangle count differs from manifest")
    objects = {name: {"bounds": bounds(values), "triangles": as_triangles(values)}
               for name, values in vertices_by_node.items()}
    anchors = []
    for a in manifest["anchors"]:
        local = point(local_worlds[a["node"]], a["local"])
        if max(abs(x-y) for x, y in zip(local, a["world"])) > TOL:
            raise ValueError("Anchor and node transform disagree: " + a["id"])
        anchors.append({"id": a["id"], "node": a["node"], "world_yup": point(placement, local)})
    if len(anchors) < 3:
        raise ValueError("At least three model anchors are required")
    areas = []
    for a, b, c in combinations(anchors, 3):
        x, y, z = a["world_yup"], b["world_yup"], c["world_yup"]
        area2 = abs((y[0]-x[0])*(z[2]-x[2]) - (y[2]-x[2])*(z[0]-x[0]))
        areas.append((area2, [a["id"], b["id"], c["id"]]))
    area, chosen = max(areas)
    if area < 1e-4:
        raise ValueError("Anchors are collinear in the horizontal X/Z plane")
    return objects, {"model_nodes": len(nodes), "model_mesh_nodes": len(objects), "triangles": triangles,
                     "glb_contains_context_nodes": False, "anchors": anchors,
                     "non_collinear_anchor_ids": chosen, "twice_horizontal_triangle_area_m2": area}


def query_top(surface, x, z):
    hits = surface.heights(x, z)
    return max(hits) if hits else None


def support_checks(objects, surfaces):
    terrain = surfaces["terrain"]
    checks = []
    curb_surface = Surface(objects["median_curb"]["triangles"]) if "median_curb" in objects else None
    for name in (*BASES, *GLASS):
        if name not in objects:
            raise ValueError("Expected actual model component is missing: " + name)
        obj = objects[name]
        lo, hi = obj["bounds"]["min"], obj["bounds"]["max"]
        surface = Surface(obj["triangles"])
        samples = []
        for a in (.03, .25, .5, .75, .97):
            for b in (.03, .25, .5, .75, .97):
                x, z = lo[0]+a*(hi[0]-lo[0]), lo[2]+b*(hi[2]-lo[2])
                ground = query_top(terrain, x, z)
                model_y = surface.heights(x, z)
                if ground is None or not model_y:
                    samples.append({"world_xz": [x, z], "status": "no_triangle_hit"})
                    continue
                support = ground
                if name == "median_kiosk_base":
                    # This base rests on the modelled median curb, not directly on terrain.
                    curb_top = query_top(curb_surface, x, z)
                    if curb_top is not None:
                        support = max(support, curb_top)
                bottom, top = min(model_y), max(model_y)
                samples.append({"world_xz": [x, z], "terrain_y": ground, "support_y": support,
                                "model_bottom_y": bottom, "model_top_y": top,
                                "bottom_minus_terrain_m": bottom-ground,
                                "bottom_minus_support_m": bottom-support,
                                "top_minus_support_m": top-support, "status": "sampled"})
        valid = [x for x in samples if x["status"] == "sampled"]
        if name in BASES:
            floating = [s for s in valid if s["bottom_minus_support_m"] > .02]
            buried = [s for s in valid if s["top_minus_support_m"] < -.02]
        else:
            floating = []
            buried = [s for s in valid if s["bottom_minus_terrain_m"] < -.01]
        status = "sampled_ok" if len(valid) == len(samples) and not floating and not buried else "issue"
        checks.append({"node": name, "kind": "foundation" if name in BASES else "glass",
                       "world_aabb_yup": obj["bounds"], "status": status,
                       "floating_samples": len(floating), "buried_samples": len(buried),
                       "samples": samples})
    return checks


def walk_floor(surfaces, x, z):
    terrain = query_top(surfaces["terrain"], x, z)
    if terrain is None:
        return None
    candidates = [(terrain, "terrain")]
    for name in WALK_LAYERS[1:]:
        y = query_top(surfaces[name], x, z)
        # Disallow an unrelated elevated surface from silently becoming the route.
        if y is not None and terrain-.05 <= y <= terrain+.25:
            candidates.append((y, name))
    return max(candidates)


def route_check(z, objects, surfaces):
    start, end, radius, height, spacing = -428., -390., .30, 1.75, .25
    cache = {}
    def profile(x):
        if x not in cache:
            points = [(x, z), (x-radius, z), (x+radius, z), (x, z-radius), (x, z+radius)]
            values = [walk_floor(surfaces, a, b) for a, b in points]
            cache[x] = None if any(v is None for v in values) else {
                "min": min(v[0] for v in values), "max": max(v[0] for v in values),
                "center": values[0][0], "center_layer": values[0][1]}
        return cache[x]

    centers = [start + i*spacing for i in range(round((end-start)/spacing)+1)]
    for x in centers:
        profile(x)
    collisions, nearest = [], []
    for name, obj in objects.items():
        low, high = obj["bounds"]["min"], obj["bounds"]["max"]
        # The swept horizontal circle is compared with each actual mesh AABB.
        dz = max(low[2]-z, 0., z-high[2])
        if high[0] < start-radius or low[0] > end+radius:
            continue
        nearest_x = min(end, max(start, (low[0]+high[0])/2))
        floor = profile(nearest_x)
        if floor and high[1] >= floor["min"]+.02 and low[1] <= floor["max"]+height:
            nearest.append({"node": name, "lateral_gap_m": dz-radius, "world_aabb_yup": obj["bounds"]})
        if dz > radius:
            continue
        reach = math.sqrt(max(0., radius*radius-dz*dz))
        left, right = max(start, low[0]-reach), min(end, high[0]+reach)
        if left > right:
            continue
        test_x = sorted({left, right, (left+right)/2, *(x for x in centers if left <= x <= right)})
        hits = []
        for x in test_x:
            floor = profile(x)
            if floor and high[1] >= floor["min"]+.02 and low[1] <= floor["max"]+height:
                hits.append(x)
        if hits:
            minus = low[2]-radius-z
            plus = high[2]+radius-z
            shift = minus if abs(minus) < abs(plus) else plus
            collisions.append({"node": name, "world_aabb_yup": obj["bounds"],
                               "potential_hit_x_interval_m": [left, right],
                               "sampled_hit_x_range_m": [min(hits), max(hits)],
                               "candidate_minimum_lateral_shift_m": shift,
                               "adjustment_note": "Move locally just beyond this AABB plus radius; recheck terrain and all other geometry."})
    central = [profile(x) for x in centers]
    missing = [x for x, p in zip(centers, central) if p is None]
    valid = [p for p in central if p is not None]
    differences = [abs(b["center"]-a["center"]) for a, b in zip(central, central[1:]) if a and b]
    widths = [p["max"]-p["min"] for p in valid]
    high_step = max(differences, default=0.)
    high_spread = max(widths, default=0.)
    issues = []
    if missing:
        issues.append("Terrain is missing below the sampled footprint")
    if high_step > .20:
        issues.append("Adjacent center samples differ by more than 0.20 m")
    if high_spread > .20:
        issues.append("The sampled 0.60 m footprint varies in height by more than 0.20 m")
    return {"world_x_range_m": [start, end], "world_z_m": z,
            "person_radius_m": radius, "person_height_m": height, "sample_spacing_m": spacing,
            "surface_selection": "Highest mapped road/sidewalk/path within [-0.05,+0.25] m of terrain; otherwise terrain. Five footprint samples.",
            "sample_count": len(centers), "missing_ground_x": missing,
            "center_surface_counts": dict(Counter(p["center_layer"] for p in valid)),
            "ground_center_y_range_m": [min(p["center"] for p in valid), max(p["center"] for p in valid)] if valid else None,
            "max_adjacent_center_change_m": high_step, "max_footprint_height_spread_m": high_spread,
            "floor_issues": issues, "collision_nodes": collisions,
            "nearest_vertical_overlapping_nodes": sorted(nearest, key=lambda n: n["lateral_gap_m"])[:8],
            "status": "obstructed_or_ground_issue" if collisions or issues else "no_hits_in_sampled_model",
            "scope": "Static exported gate pose and mapped ground only; AABBs are conservative and this is not a physics or real-world access test."}


def verify(package, context_path):
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    context, surfaces = load_context(context_path)
    issues = []
    placement, placement_result = placement_for(package, manifest, context["origin_yup"], issues)
    objects, model = inspect_model(package, manifest, placement)
    support = support_checks(objects, surfaces)
    routes = [route_check(z, objects, surfaces) for z in (-63., -45.)]
    if any(s["status"] != "sampled_ok" for s in support):
        issues.append("Foundation or glass sampled grounding issue")
    if any(r["status"] != "no_hits_in_sampled_model" for r in routes):
        issues.append("A sampled candidate route has an obstacle or ground issue")
    status = "failed" if issues else "passed_with_limits" if placement_result["status"] == "verified" else "inconclusive"
    return {"status": status, "issues": issues,
            "manifest_sha256": sha256(package / "manifest.json"), "context_sha256": sha256(context_path),
            "placement": placement_result, "model": model,
            "grounding": {"status": "sampled_ok" if all(s["status"] == "sampled_ok" for s in support) else "issue",
                          "method": "5 x 5 interior footprint samples against actual component triangles and context terrain; kiosk base also checks the modelled curb as support.",
                          "floating_threshold_m": .02, "glass_burial_tolerance_m": .01,
                          "components": support},
            "routes": routes,
            "limitations": ["Static pose and conservative mesh AABBs; no simulation of interactions, moving barriers, dynamics or collisions with omitted campus objects.",
                            "Terrain and surface samples reflect an approximate existing project, not measurements of the real campus.",
                            "Finite sampling cannot prove every point in a footprint or route is clear or connected.",
                            "No old-road replacement, context integration, real-world public access or accessibility is validated."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.package.resolve(), args.context.resolve())
    except Exception as exc:
        result = {"status": "failed", "error": str(exc), "error_type": type(exc).__name__}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Lide site check:", result["status"], "report:", args.report)
    if result.get("error"):
        print(result["error_type"] + ":", result["error"])
    return 0 if result["status"] == "passed_with_limits" else 2 if result["status"] == "inconclusive" else 1


if __name__ == "__main__":
    sys.exit(main())
