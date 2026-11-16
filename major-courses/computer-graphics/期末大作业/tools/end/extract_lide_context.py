"""Extract the existing Lide gate surroundings without rebuilding the campus.

Run from the project root with build/modeling-pipeline/venv/Scripts/python.exe.
The output is a read-only modelling reference, not a replacement ground mesh.
"""

import hashlib
import json
import math
import struct
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "main/end/code/assets"
OUT = ROOT / "main/end/modeling/assets/lide-gate/context"
BOUNDS = (-438.0, -378.0, -82.0, -22.0)  # world xmin, xmax, zmin, zmax
ORIGIN_XZ = (-410.0, -54.0)

# Exact draw bindings and colour multipliers from main/end/code/main.cpp::world.
# The campus road markings use white.ppm, which is uniformly RGB 255; a flat
# colour gives precisely the same base colour without copying that texture.
LAYERS = [
    ("terrain", "campus-terrain", "campus-ground", (1, 1, 1), "terrain"),
    ("road-surface", "campus-road-surface", "campus-asphalt", (1, 1, 1), "vehicle_surface"),
    ("road-markings", "campus-road-markings", None, (.92, .93, .88), "road_markings"),
    ("road-curbs", "campus-road-curbs", "concrete", (.88, .89, .85), "road_curb"),
    ("network-sidewalk", "campus-network-sidewalk", "campus-sidewalk", (1, 1, 1), "sidewalk"),
    ("network-paths", "campus-network-paths", "paving", (.97, .96, .92), "pedestrian_path"),
    ("lide-walk", "campus-lide-walk", "qiaoyuan-brick", (1, 1, 1), "gate_brick_paving"),
    ("lide-walk-edge", "campus-lide-walk-edge", "concrete", (.84, .83, .78), "gate_paving_edge"),
    ("corridor-bed", "campus-lide-corridor-bed", "grass", (.54, .60, .39), "planting_bed"),
    ("corridor-curbs", "campus-lide-corridor-curbs", "concrete", (.86, .85, .80), "planting_curb"),
    ("network-verges", "campus-network-verges", "grass", (.67, .82, .60), "road_verge"),
]
GATE_PARTS = ("stone", "metal", "glass", "roof", "sign")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_and_hash(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def read_mesh(path):
    data = path.read_bytes()
    if len(data) < 8 or data[:4] != b"SZM1":
        raise ValueError(f"Invalid SZM1 header: {path}")
    count = struct.unpack_from("<I", data, 4)[0]
    if count % 3 or len(data) != 8 + 32 * count:
        raise ValueError(f"Invalid SZM1 size or triangle count: {path}")
    for values in struct.iter_unpack("<24f", memoryview(data)[8:]):
        yield [values[:8], values[8:16], values[16:24]]


def clip_plane(poly, axis, bound, keep_greater):
    """Clip a polygon to one X/Z half-plane, interpolating xyz, normal and UV."""
    if not poly:
        return []
    result = []
    prev = poly[-1]
    prev_inside = prev[axis] >= bound if keep_greater else prev[axis] <= bound
    for cur in poly:
        inside = cur[axis] >= bound if keep_greater else cur[axis] <= bound
        if inside != prev_inside:
            t = (bound - prev[axis]) / (cur[axis] - prev[axis])
            result.append(tuple(a + t * (b - a) for a, b in zip(prev, cur)))
        if inside:
            result.append(cur)
        prev, prev_inside = cur, inside
    return result


def area2(a, b, c):
    # Use 3D area, not projected X/Z area: curb side faces can be vertical.
    ab = [b[i] - a[i] for i in range(3)]
    ac = [c[i] - a[i] for i in range(3)]
    cross = (ab[1] * ac[2] - ab[2] * ac[1],
             ab[2] * ac[0] - ab[0] * ac[2],
             ab[0] * ac[1] - ab[1] * ac[0])
    return sum(v * v for v in cross)


def extract(path):
    xmin, xmax, zmin, zmax = BOUNDS
    result = []
    candidate_count = 0
    for tri in read_mesh(path):
        if (max(v[0] for v in tri) < xmin or min(v[0] for v in tri) > xmax or
                max(v[2] for v in tri) < zmin or min(v[2] for v in tri) > zmax):
            continue
        candidate_count += 1
        poly = tri
        for axis, bound, greater in ((0, xmin, True), (0, xmax, False),
                                     (2, zmin, True), (2, zmax, False)):
            poly = clip_plane(poly, axis, bound, greater)
        for i in range(1, len(poly) - 1):
            clipped = (poly[0], poly[i], poly[i + 1])
            if area2(*clipped) > 1e-16:
                result.append(clipped)
    if not result:
        raise ValueError(f"No triangles inside bounds: {path}")
    return result, candidate_count


def sample(triangles, x, z):
    """Sample non-vertical triangles in their own X/Z coordinates; no extrapolation."""
    heights = []
    for a, b, c in triangles:
        det = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2])
        if abs(det) < 1e-10:
            continue
        u = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / det
        v = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / det
        w = 1.0 - u - v
        if min(u, v, w) >= -1e-7:
            heights.append(u * a[1] + v * b[1] + w * c[1])
    if not heights:
        raise ValueError(f"No surface at ({x}, {z})")
    return max(heights)


def write_mesh(path, triangles, origin, ground_crop=None):
    with path.open("wb") as stream:
        stream.write(struct.pack("<4sI", b"SZM1", 3 * len(triangles)))
        for tri in triangles:
            for vertex in tri:
                x, y, z, nx, ny, nz, u, v = vertex
                if ground_crop:
                    # Coordinates are based on pixel *edges*, preserving linear
                    # base-level sampling after the terrain texture is cropped.
                    left, bottom, width, height, source_w, source_h = ground_crop
                    u = (u * source_w - left) / width
                    v = (v * source_h - bottom) / height
                stream.write(struct.pack("<8f", x-origin[0], y-origin[1], z-origin[2],
                                         nx, ny, nz, u, v))


def write_texture(name, terrain_triangles):
    source = ASSETS / f"{name}.ppm"
    output = OUT / "textures" / f"{name}.png"
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    # OpenGL consumes the first PPM row as the bottom row. PNG stores its first
    # row at the top, so reverse rows for Blender/Three.js to preserve UVs.
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    width, height = image.size
    crop_mapping = None
    if name == "campus-ground":
        vertices = [v for tri in terrain_triangles for v in tri]
        umin, umax = min(v[6] for v in vertices), max(v[6] for v in vertices)
        vmin, vmax = min(v[7] for v in vertices), max(v[7] for v in vertices)
        # Two original pixels of margin keep linear sampling away from the crop edge.
        left = max(0, math.floor(umin * width) - 2)
        right = min(width, math.ceil(umax * width) + 2)
        bottom = max(0, math.floor(vmin * height) - 2)
        top = min(height, math.ceil(vmax * height) + 2)
        image = image.crop((left, height-top, right, height-bottom))
        crop_mapping = (left, bottom, right-left, top-bottom, width, height)
    image.save(output, format="PNG", compress_level=9)
    return crop_mapping, {
        "path": output.relative_to(OUT).as_posix(),
        "sha256": sha256(output),
        "source": path_and_hash(source),
        "source_dimensions": [width, height],
        "output_dimensions": list(image.size),
        "source_pixels": "PPM first stored row is uploaded as OpenGL bottom row",
        "png_pixels": "vertical row reversal; PNG top row is source final row",
        "wrap": "repeat",
        "base_level_resampling": "none",
        "crop_left_bottom_width_height": list(crop_mapping[:4]) if crop_mapping else None,
        "mipmaps": "regenerated by the consumer; cropped terrain mip levels may differ from the full texture",
    }


def clip_self_test():
    # A crossing face exercises intersection, interpolation and preservation of
    # vertical faces; vertices/UVs here deliberately encode their X/Z values.
    polygon = [(-2, 0, 0, 0, 1, 0, -2, 0), (2, 0, 0, 0, 1, 0, 2, 0),
               (0, 2, 0, 0, 1, 0, 0, 0)]
    polygon = clip_plane(polygon, 0, -1, True)
    polygon = clip_plane(polygon, 0, 1, False)
    if not all(-1 <= v[0] <= 1 and abs(v[0] - v[6]) < 1e-12 for v in polygon):
        raise AssertionError("Clip interpolation failed")
    if not any(area2(polygon[0], polygon[i], polygon[i+1]) > 0 for i in range(1, len(polygon)-1)):
        raise AssertionError("Vertical triangle was lost")


def main():
    clip_self_test()
    # The untextured markings below are only equivalent while WHITE stays white.
    with Image.open(ASSETS / "white.ppm") as white:
        if white.convert("RGB").getextrema() != ((255, 255),) * 3:
            raise ValueError("white.ppm is no longer uniformly white")
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    (OUT / "textures").mkdir(parents=True, exist_ok=True)
    extracted = {}
    counts = {}
    for layer_id, source_name, _, _, _ in LAYERS:
        extracted[layer_id], counts[layer_id] = extract(ASSETS / f"{source_name}.mesh")
    origin = [ORIGIN_XZ[0], sample(extracted["terrain"], *ORIGIN_XZ), ORIGIN_XZ[1]]

    texture_names = sorted({texture for _, _, texture, _, _ in LAYERS if texture})
    textures = {}
    ground_crop = None
    for name in texture_names:
        crop, textures[name] = write_texture(name, extracted["terrain"])
        if name == "campus-ground":
            ground_crop = crop

    layers = []
    for layer_id, source_name, texture, color, role in LAYERS:
        output = OUT / "meshes" / f"{layer_id}.mesh"
        triangles = extracted[layer_id]
        write_mesh(output, triangles, origin, ground_crop if layer_id == "terrain" else None)
        layers.append({
            "id": layer_id, "mesh": output.relative_to(OUT).as_posix(),
            "mesh_sha256": sha256(output), "texture": textures[texture]["path"] if texture else None,
            "color": list(color), "role": role, "writable": False,
            "source_mesh": path_and_hash(ASSETS / f"{source_name}.mesh"),
            "source_texture": path_and_hash(ASSETS / f"{texture or 'white'}.ppm"),
            "source_candidate_triangles": counts[layer_id], "triangles": len(triangles),
            "note": "WHITE is exactly RGB (255,255,255); constant color is equivalent" if texture is None else None,
        })

    anchor_specs = [
        ("west-vehicle", -438.0, -54.0, "terrain", [-1, 0, 0],
         "Road is painted into campus-ground on the terrain here."),
        ("east-vehicle", -378.0, -54.0, "road-surface", [1, 0, 0],
         "Candidate on existing vehicle surface at the context edge."),
        ("east-sidewalk", -378.0, -64.0, "network-sidewalk", [1, 0, 0],
         "Candidate on existing sidewalk at the context edge."),
    ]
    anchors = []
    for name, x, z, layer, direction, note in anchor_specs:
        world = [x, sample(extracted[layer], x, z), z]
        anchors.append({"id": name, "world_yup": world,
                        "local_yup": [world[i] - origin[i] for i in range(3)],
                        "sample_layer": layer, "outward_yup": direction,
                        "status": "candidate; routing and clearance require integration verification",
                        "note": note})

    gate_sources = [ASSETS / f"campus-gate-lide-{part}.mesh" for part in GATE_PARTS]
    source_files = [ROOT / "main/end/code/main.cpp", ASSETS / "campus-gates.tsv",
                    ASSETS / "campus-gates.json", ASSETS / "CREDITS.txt", Path(__file__).resolve()]
    source_files += gate_sources + [ASSETS / "white.ppm"]
    meta = {
        "schema_version": 1,
        "description": "Read-only local snapshot of the existing approximate campus scene; excludes the old Lide gate body.",
        "units": "m", "coordinate_system": "RH_Y_UP", "writable": False,
        "origin_yup": origin,
        "origin_calculation": "At world X=-410, Z=-54, barycentric interpolation of the rendered campus-terrain.mesh triangle; no camera-height offset.",
        "coordinates": {"mesh_vertices": "local Y-up = source campus world Y-up - origin_yup",
                        "world_from_mesh": "local_yup + origin_yup",
                        "blender_z_up_from_local": "(x,y,z)_Yup -> (x,-z,y)_Blender",
                        "axes": "X east, Y up, Z south; right-handed"},
        "bounds_world": {"x": [BOUNDS[0], BOUNDS[1]], "z": [BOUNDS[2], BOUNDS[3]]},
        "clip": "Each intersecting source triangle is polygon-clipped to the X/Z rectangle; xyz, normals and UV are linearly interpolated. Degenerate 3D faces are discarded.",
        "height_sampling": {"mesh": "meshes/terrain.mesh", "method": "Barycentric interpolation in local X/Z on nonvertical triangles; add origin_yup[1] for world height. Do not extrapolate.",
                            "note": "This is rendered terrain, not necessarily the uppermost walk/road surface or a surveyed elevation."},
        "layers": layers, "textures": textures, "anchors": anchors,
        "replacement_scope": {
            "old_gate": {"index": "main/end/code/assets/campus-gates.tsv", "index_id": "lide",
                         "action_for_future_integration": "Suppress exactly the five complete L rows with id=lide; no shared layer is disabled by this declaration.",
                         "materials": list(GATE_PARTS), "source_meshes": [path_and_hash(p) for p in gate_sources]},
            "context_rectangle_is_not_a_deletion_mask": True,
            "ground_and_shared_layers": "No ground, roadway, pavement or curb deletion is authorized by this snapshot. Any later replacement must define its exact local footprint, clip shared meshes and handle the base texture before integration.",
            "baked_road": "The old terrain texture contains the western gate vehicle surface and some markings; suppressing only campus-road-surface.mesh does not erase them."
        },
        "omitted": ["The five old gate-body meshes are intentionally excluded to avoid duplicate gate geometry.",
                    "Vegetation geometry, photographs and unrelated buildings are excluded.",
                    "campus-lide-corridor-paving.mesh and -lawn.mesh, and campus-network-stairs.mesh have no triangles in this rectangle."],
        "provenance": {"sources": [path_and_hash(p) for p in source_files],
                       "attribution": "Existing campus CREDITS.txt: OpenStreetMap contributors / ODbL; terrain from Mapzen Terrain Tiles with SRTM/GMTED2010; grass colour map ambientCG Grass004 / CC0. The generated campus-ground derives in part from this grass map.",
                       "references": "Existing scene is a photo-informed visual approximation, not a land survey; no campus reference photos are included."},
        "reproduce": "build/modeling-pipeline/venv/Scripts/python.exe tools/end/extract_lide_context.py"
    }
    (OUT / "context.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(layers)} layers to {OUT}")
    print("origin_yup:", origin)
    for anchor in anchors:
        print(anchor["id"], anchor["world_yup"], anchor["sample_layer"])


if __name__ == "__main__":
    main()
