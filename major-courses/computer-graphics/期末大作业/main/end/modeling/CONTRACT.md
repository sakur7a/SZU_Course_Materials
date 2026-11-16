# M0 asset contract (v1)

This file is the shared implementation contract for the first fixture. The fixture is synthetic, not a reconstruction of a real campus building. Parent design: ../PIPELINE.md.

## Package

An output directory contains `model.blend`, `model.glb`, `manifest.json`, `legacy/`, and optional `previews/`. All runtime paths are relative to this output directory. No external texture URI is allowed in GLB.

Manifest fields:

```json
{
  "schema_version": 1,
  "asset_id": "pipeline-fixture",
  "asset_version": "0.1.0",
  "name": "M0 · 建模管线测试场景",
  "synthetic": true,
  "units": "m",
  "coordinate_system": "RH_Y_UP",
  "model": "model.glb",
  "bounds": {"min": [-4,0,-3], "max": [4,4,3]},
  "nodes": [{"id":"root","parent":null,"local_matrix":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],"roles":["visual"]}],
  "materials": [{"id":"stone","base_color":[1,1,1,1],"alpha_mode":"OPAQUE","alpha_cutoff":0.5,"double_sided":false,"legacy_texture":"legacy/textures/stone.rgba"}],
  "primitives": [{"id":"platform","node":"root","mesh":"legacy/meshes/platform.mesh","material":"stone"}],
  "anchors": [{"id":"entry","node":"root","local":[0,0,0],"world":[0,0,0]}],
  "interaction": {"node":"hinge","axis":[0,1,0],"range_deg":[0,90]},
  "hierarchy_probe": {"parent":"building","descendant":"frame","rotation_deg":15},
  "views": {"hero":{"eye":[9,7,11],"target":[0,1.5,0]},"front":{"eye":[0,3,13],"target":[0,1.5,0]},"top":{"eye":[0,15,0.01],"target":[0,0,0]}},
  "statistics": {"triangles":0,"meshes":0},
  "files": {}
}
```

Values above illustrate the structure; exporter computes actual geometry and bounds. `files` maps relative output paths to SHA-256 (exclude manifest itself and validation reports). The parent orchestrator adds provenance/input hashes and validates schema.

Matrices are 16 floats, column-major, column vectors. `world = parent_world * local`. All manifest/legacy coordinates are right-handed Y-up metres. Blender is Z-up; basis conversion is `(x,y,z) -> (x,z,-y)`, and local matrices transform by basis conjugation. Blender's GLB exporter performs its own Y-up conversion, so do not pre-convert the scene. Stable node IDs are Blender object names and exported glTF node names. Do not flatten hierarchy or apply matrices twice.

Interactions apply `node.local_matrix * rotation(axis, angle)` after loading. Angles in user-facing controls are degrees. Hierarchy probe rotates its parent's local Y axis by 15 degrees, then restores it. An implementation must compare a descendant's world transform, not merely count ancestors.

## Legacy (UTF-8 TSV, comments begin with #)

- `legacy/nodes.tsv`: `N<TAB>id<TAB>parent-or--<TAB>comma-separated-roles<TAB>16 column-major floats separated by TAB`.
- `legacy/materials.tsv`: `T<TAB>id<TAB>texture-relative-to-package-or--<TAB>alpha-mode<TAB>alpha-cutoff<TAB>double-sided-0-or-1<TAB>r<TAB>g<TAB>b<TAB>a`.
- `legacy/primitives.tsv`: `P<TAB>primitive-id<TAB>node-id<TAB>material-id<TAB>mesh-relative-to-package`.
- `legacy/anchors.tsv`: `A<TAB>id<TAB>node-id<TAB>local-x<TAB>local-y<TAB>local-z<TAB>expected-world-x<TAB>expected-world-y<TAB>expected-world-z`.
- `legacy/controls.tsv`: `I<TAB>node<TAB>axis-x<TAB>axis-y<TAB>axis-z<TAB>min-deg<TAB>max-deg`; `H<TAB>parent<TAB>descendant<TAB>rotation-deg`; `V<TAB>name<TAB>eye-x<TAB>eye-y<TAB>eye-z<TAB>target-x<TAB>target-y<TAB>target-z`.
- SZM1 mesh: magic `SZM1`, little-endian uint32 vertex count, then triangles expanded as 8 little-endian float32 values per vertex: position xyz, normal xyz, UV uv. Vertices are node-local in converted basis; modifiers are evaluated. Normal transforms at runtime use inverse-transpose node world matrix.
- Texture: little-endian uint32 width and height followed by bottom-up RGBA8 pixels, matching OpenGL UV origin. Texture RGB is sRGB; alpha is linear. Use GL_SRGB8_ALPHA8 in C++ and a consistent output transfer. GLB uses embedded PNG and the standard exporter handles its UV/image convention.

M0 needs a visible textured pavilion, a true four-level parent chain, nontrivial parent transform, one hinged movable element, an actual MASK cutout, three non-collinear anchors, and a ground plane. A decorative display of a scene graph alone does not satisfy these checks.
