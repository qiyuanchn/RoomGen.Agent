# blender_agent

Headless-Blender scene generator that produces PanoWorld-format viewpoints
(equirect panorama + Euclidean depth) from a JSON scene spec.  P0 built the
shell; P1 dresses it as a furnished, textured home lit by a physical sky; P3
replays official **3D-FRONT** houses (dataset shell + 3D-FUTURE models).

## One command

```bash
bash scripts/blender_agent_run.sh [spec] [outdir]
```

| spec | what it builds |
|---|---|
| `configs/blender_agent_p0_demo.json` (default) | bare two-room shell (P0 regression) |
| `configs/blender_agent_p1_furnished.json` | same shell, furnished + textured + lit (P1) |

Default output is `outputs/blender_agent/demo_house`; the P1 run writes
`outputs/blender_agent/furnished_home`.  The script runs six stages:

1. ERP orientation self-test (`render_selftest.py`, its own Blender process)
2. Poly Haven asset libraries (`run_build.py --prepare-assets-only`)
3. render (`run_build.py --no-denoise`: colour PNG + Position-pass EXR per camera)
4. OpenImageDenoise pass over the linear EXR (`run_build.py --denoise-only`)
5. host-side export (`export_scene.py`)
6. validation (`tools/validate_blender_agent_p0.py`) → `validation_report.json`

Useful environment overrides: `PANOWORLD_SAMPLES` (default 128),
`PANOWORLD_RES` (default `2048x1024`, e.g. `1024x512` for a fast look),
`PANOWORLD_CACHE`, `PANOWORLD_TEXRES` (`1k`/`2k` furniture textures),
`PANOWORLD_BLENDER`, `PANOWORLD_HOSTPY`, and `PANOWORLD_EXPORT_GLB=1` to also
write `build/scene.glb` for the browser 3D viewer.

## 3D-FRONT replay (P3)

```bash
bash scripts/front3d_replay.sh <3D-FRONT/uid.json> <3D-FUTURE-model> <outdir> [3D-FRONT-textures]
```

Replays a real dataset house through the same stages as above: the shell comes
from the scene JSON's **inline mesh triangles**, the furniture from the
3D-FUTURE models at their dataset poses, and the viewpoints are synthesized (one
pano per room at eye height, or the dataset's own `extension.snapshots` with
`PANOWORLD_CAMERAS=snapshots`).

```bash
# parse-only summary of any scene JSON (no Blender, no dataset download of models)
python blender_agent/front3d.py --scene 3D-FRONT/<uid>.json --model-root 3D-FUTURE-model --dry-run
# write the bridged spec both entry points share
python blender_agent/front3d.py --scene ... --model-root ... --spec-out /tmp/front3d_spec.json
```

Both entry points are equivalent: `run_build.py --front3d-scene ... --front3d-model-root ...`
parses and bridges in memory, and `run_build.py --spec <bridged spec>` re-uses a
spec written earlier (the build directory always records the spec it used as
`build/front3d_scene_spec.json`).

What the importer does, and why (the full format truth is
`blender_agent/front3d.py`'s docstring):

* **frame**: the dataset is Y-up metres; the replay converts with
  `M = Rx(+90°)`, `(x, y, z) -> (x, -z, y)`.  A *rotation*, not BlenderProc's
  y-flip, so triangle winding (and therefore normals) survives.
* **furniture**: `v_world = R(q)·(s ⊙ v_model) + pos` — per-axis scale, then the
  `(x, y, z, w)` quaternion, then the translation, applied to raw Y-up model
  vertices.  `pos` is the model's base centre (3D-FUTURE models span
  `y ∈ [0, height]`), so nothing is re-anchored and the dataset pose is exact.
* **spec bridge**: rooms and cameras become a normal v2 spec (room outlines are
  recovered from the shell's `Floor` meshes, which reproduces the dataset's own
  `room.size` area to ~1 %); furniture items keep their non-uniform
  `scale_xyz`.  Items the dataset flags `valid: false`, items with no
  `furniture[]` entry, and items outside every room outline are reported in the
  `front3d` block instead of producing an invalid spec.
* **partial data**: 3D-FUTURE is only partly public — missing model folders are
  skipped with a report (`manifest.front3d.furniture_not_on_disk`), exactly as
  the dataset's own loaders do.
* **lighting**: `Window` meshes get real glass and each one carries an area
  light inside the reveal (the dataset has no `windows` entries for the P0/P1
  portal code to use); ceilings are *not* made emissive, unlike BlenderProc.

Acceptance test (hand-built two-room fixture, no dataset needed):

```bash
bash tests/test_front3d_replay.sh          # 1024x512 @ 32 samples, ~1.5 min
PANOWORLD_RES=2048x1024 bash tests/test_front3d_replay.sh   # canonical size
```

## Modules

| file | role |
|---|---|
| `spec.py` | schema + validation (pure stdlib, shared with the host tools), format constants (`R0`, depth scale), wall-line helpers |
| `shell.py` | room polygons → floors/ceilings/walls, openings cut with Boolean DIFFERENCE |
| `details.py` | skirting boards, door/window casings, sill boards, window glass, shell bevels, props (rug, table lamp) |
| `materials.py` | Poly Haven PBR sets → Principled with physically-scaled box mapping |
| `assets.py` | Poly Haven downloader (host) + glTF import/normalise/library/instancing (Blender) |
| `front3d.py` | host: 3D-FRONT scene JSON parse, transform math, `scene_spec` bridge; Blender: dataset shell + 3D-FUTURE model import, window lights |
| `stage.py` | materials, Nishita sky world, sun, enclosing sky dome, window portal lights, lamp lights |
| `panorama.py` | equirect camera, Position-pass depth, strip/binary PLY export, OIDN pass |
| `render_selftest.py` | renders emissive markers along known world axes and checks the ERP mapping |
| `settle.py` | physics settling: rigid bodies + drop-to-rest, bakes the settled transforms (see below) |
| `export_glb.py` | Blender-side GLB export for the browser 3D viewer (`run_build.py --export-glb`): `scene.glb` + `scene3d_manifest.json` |
| `run_build.py` | Blender entry point (build, render, denoise, asset prep, 3D-FRONT replay, GLB export) |
| `export_scene.py` | host: build output → `scene/viewpoints/<cam>/*`, `map.json`, `topdown.png` |

## Output layout

```
<outdir>/build/            pano_<cam>.png, depth_<cam>.npy, position_*.exr,
                           raw_*.exr (linear, for the denoise pass),
                           shell.ply (shell only), furnished.ply (all visible
                           geometry: shell + details + furniture + sky dome),
                           scene.glb + scene3d_manifest.json (with
                           --export-glb, for tools/scene3d_viewer/),
                           build_manifest.json, selftest_report.json
<outdir>/scene/viewpoints/<cam>/{panoImage_2048.png, place_depth.png,
                           place_depth_scale.txt, extrinsics.txt}
<outdir>/scene/{map.json, topdown.png, export_report.json}
<outdir>/validation_report.json
```

`scene.glb` holds the shell, details and furniture with their PBR materials in
glTF's +Y-up metres; the render-only sky dome and the cameras/lights are left
out. `--skip-render --export-glb` is the fast (geometry only) path, and
`bash scripts/scene3d_view.sh <outdir>` serves the result in the browser - see
"Viewing scenes in the browser" in the top-level README.

`furnished.ply` is the ground truth for the depth check: the validator raycasts
every ERP ray against it and compares with `place_depth.png / scale`
(valid ≥ 99.5 %, median |Δ| < 5 mm, p95 < 2 cm).  Doors are checked separately
and per opening: `shell.ply` is measured with horizontal rays cast from both
sides of every wall, so each doorway has to be really open, really surrounded by
wall, and as wide and as high as the spec says (`door_openings_geometry`).
Which doorways a camera sees through is projected into the ERP frame with the
camera matrix inverse (`world_to_erp`) and reported as
`door_visibility_coverage` -- an informational metric: reasonable occlusion is a
number there, not a scene failure.

## Assets (Poly Haven, CC0)

```bash
python blender_agent/assets.py list --category seating          # browse
python blender_agent/assets.py fetch --assets ph:sofa_02         # cache 2k
python blender_agent/assets.py fetch --assets ph:sofa_02 --resolution 1k
python blender_agent/assets.py check --assets ph:sofa_02         # offline
blender --background --factory-startup \
  --python blender_agent/assets.py -- preview --assets ph:sofa_02 ph:oak_wood_planks \
  --out /tmp/sheet.png                                            # scale + facing
```

Cache: `third_party/assets/polyhaven/` (gitignored) — `models/<id>/`,
`textures/<id>/`, `library/<id>.<res>.blend`, `index.json`.

Conventions applied by the loader (`asset_meta.json` holds per-asset yaw):

* metres, verified against the API bounding box (a wild scale raises);
* origin at the **bottom-centre** of the bbox, so `z = 0` stands on the floor;
* the **front faces -Y**; `rot_z_deg` in a spec turns from there
  (0 = -Y, 90 = +X, 180 = +Y, 270 = -X);
* repeats are *collection instances* (mesh + textures shared);
* textures are built from the **1k** variant by default because the render-time
  texture cache is the biggest memory item.

## Scene spec (v2)

P0 fields plus optional blocks (a P0 spec keeps working unchanged):

```jsonc
"style":     {"sun_elevation_deg": 35, "sun_azimuth_deg": 210, "sky_strength": 1.0,
              "sun_energy": 3.0, "view_transform": "AgX", "exposure": 1.4},
"materials": {"floor": "ph:oak_wood_planks", "wall": "ph:beige_wall_001",
              "rug": "ph:poly_wool_herringbone", "floor_tile_m": 1.2,
              "normal_strength": {"floor": 0.45, "wall": 0.7, "rug": 0.3}},
"details":   {"skirting": true, "casings": true, "bevels": true, "glass": true},
"lighting":  {"window_portal_w": 42.0, "lamp_w": 8.0, "portal_margin": 0.02},
"furniture": [{"asset": "ph:sofa_02", "room": "living", "pos": [3.0, 3.63],
               "rot_z_deg": 0, "mount": "floor", "z": null, "scale": 1.0},
              {"asset": "builtin:rug", "room": "living", "pos": [3.05, 2.5],
               "size": [2.6, 1.5]},
              {"asset": "ph:modern_ceiling_lamp_01", "room": "living",
               "pos": [3.1, 2.7], "mount": "ceiling"}]
```

`mount: "ceiling"` lifts an asset so its top touches the ceiling;
`builtin:rug` / `builtin:table_lamp` are generated props (`size` = [length, width]),
and the functional `builtin:` furniture of `blender_agent/fixtures.py` (kitchen
run, bathroom set, desk, dining set, wardrobe, ...) is placed the same way -- its
footprint is the box union the builder emits, so `checks.py` sizes exactly what
the render shows.

The P3 replay extends the same schema (a P0/P1 spec keeps working unchanged):

```jsonc
"front3d":   {"scene_json": "/data/3D-FRONT/<uid>.json", "model_root": "/data/3D-FUTURE-model",
              "texture_root": "/data/3D-FRONT-textures", "shell": "shell_mesh"},
"furniture": [{"asset": "front3d:<jid>", "room": "BedRoom-151355", "pos": [5.4, -1.7],
               "z": 0.0, "rot_z_deg": 90.0, "scale_xyz": [1.0, 0.8, 1.1]}]
```

## Layout generation: topology families and functional furniture

`blender_agent/layout_gen.py` turns a seed into a furnished spec.  The shell is
a **topology family** -- a small room pattern with seed-drawn cell sizes, so the
same family gives a different flat for every seed:

| `--shape` | plan |
|---|---|
| `row` (default) | the railroad flat: 2-7 rooms in one row along +X |
| `block` | the same chain wrapped into two rows, an L / block outline |
| `compact_2x2` | 4 rooms in a 2x2 block (living with the kitchen beside it, bedroom and bathroom behind) |
| `l_shape` | an L: dining and kitchen together, living below, bathroom off it, bedroom at the end |
| `central_living` | the living room spans the middle band, rooms open off it north and south (radiating) |
| `corridor_single` | a hall strip along one side, rooms over it, living closing the end |
| `corridor_double` | rooms on both sides of a middle hall |
| `loop` | a circulation ring (living -> hall -> kitchen -> dining -> living) with bedroom and bathroom as leaves |
| `auto` | the seed draws the room count and one of the families (`PANOWORLD_LAYOUT_SHAPE=auto bash scripts/layout_gen_run.sh <seed> <out>`) |

Rooms are rectangles (a family's outline can be an L); the functional programme
covers living room, bedroom(s), kitchen, bathroom plus dining, study, entry,
corridor, storage, laundry and balcony.  The room *type* of a cell follows the
drawn size, so a small service cell becomes a bathroom and never a 3 m2 bedroom.

**Semantics** (enforced, and reported in `layout_report.json` -> `door_semantics`):

* doors come from a spanning tree over the shared walls scored by function: a
  corridor serves the rooms, the kitchen opens onto the dining room, and a
  bedroom is never entered from a kitchen (`DOOR_CLASS_SCORE`);
* a bathroom has exactly one door -- it is entered, never crossed;
* with all bedrooms and bathrooms removed the remaining (public) rooms are still
  connected, so a bedroom is never the public way through the flat (an en-suite
  bathroom hanging off its bedroom is fine);
* windows follow the real exterior wall spans *and* the room function: living /
  bedroom / study / dining get 1-2, kitchens nearly always one, bathrooms and
  laundries rarely, corridors / storage / entry none.

**Functional furniture** (`blender_agent/fixtures.py`, spec assets
`builtin:<part>`): a kitchen counter run (cabinet modules, stove, sink) with a
fridge, wall cupboards above it; a bathroom set (toilet, washbasin, shower
cabin); desk, chair and bookshelf for the study; dining table with chairs;
wardrobe, shoe cabinet, storage shelving and a washing machine.  Each fixture
claims the clear space its use needs -- the fridge door swing, the cook's
standing spot, the pulled-out desk chair (`usage_m` in the layout report) -- and
the layout refuses to place blocking furniture there.  A windowless bathroom or
hall gets a ceiling lamp, because a door alone leaves those rooms black.

```bash
python blender_agent/layout_gen.py --seed 3 --shape loop --out /tmp/spec.json
python blender_agent/layout_gen.py --seed 3 --shape auto --rooms 5 --out /tmp/spec.json
python blender_agent/checks.py /tmp/spec.json          # placement + circulation
```

## Furniture settling (drop to rest)

Placement in a spec is *analytic*: a piece's bottom goes to the authored `z`
(0 = "on the floor") and is only cross-checked against the room outline. Nothing
says the piece really stands on the slab, and two pieces may well be authored
into each other. `blender_agent/settle.py` closes that gap: right after
`place_furniture` every furniture instance gets a rigid body, the shell
(floor/walls/ceiling) becomes a passive collider, the pieces are dropped a
couple of centimetres and simulated until they rest, then the settled pose is
**baked** into the object transforms -- BlenderProc's drop-to-rest recipe
(rigid bodies + `bpy.ops.ptcache` stepping + baking), with nothing of the
simulation left in the scene the cameras see.

What that means in practice:

* the manifest carries the *settled* geometry: `furniture[i].bbox_z` is the
  world-space bottom/top after settling, and `furniture[i].settle` records the
  mode, the reasoning, the values before/after, the drop and the movement. Every
  floor-resting piece of the shipped P1 spec ends up 0-0.1 mm above the floor
  (the collision margin *is* the resting gap);
* a piece that was authored into something else is nudged out first
  (minimum translation over the hulls) instead of being launched -- the P1 sofa
  slides 4.6 cm out of the bookshelf's hull, the bedroom plant 6.3 cm out of the
  east wall;
* pieces gravity does not support keep their authored transform *exactly*:
  `mount: "ceiling"`, `mount: "wall"`, and anything with an authored `z` above
  `lift_tolerance_m` (default 2 cm) -- the book stacks on the shelf, the pictures
  on the wall, the lamp on the nightstand. They still get their (kinematic) rigid
  body, in a collision group of their own, so they neither fall nor push;
* thin slabs (`builtin:rug`, <= `flat_height_m`, default 3 cm) share a collision
  group with the floor instead of the furniture: a 12 mm rug is floor finish, it
  does not lift the sofa standing on it.

The three implementation details that make it work on *these* models are worth
knowing when something looks off:

1. **hull proxies.** Furniture models are collection instances and Blender's
   rigid-body operator only accepts meshes, so each placement is simulated as
   the convex hull of its world vertices, and the proxy's rigid transform is
   composed onto the object afterwards (`object.matrix_world = P @ before`); the
   instanced meshes and their textures are never touched.
2. **the body's origin is its centre of mass** (that is what Bullet takes the
   object origin for): it is placed over the piece's *floor contact patch* (the
   geometry within `base_band_m`, default 1.5 cm, of the lowest point) and at the
   height of the hull's volume centroid. A body whose mass hangs outside its
   footprint tips over by itself -- the potted plants' canopy centre is 10 cm off
   the pot, and they rolled over within a second before this.
3. **`use_split_impulse` + margins.** Active bodies carry `margin_m` (default
   2 mm; Blender's 40 mm default visibly floats everything), the passive shell a
   zero margin, and a deeply interpenetrating pair is separated geometrically
   first, because Bullet's penetration recovery overshoots by roughly the
   overlap.

```bash
# on by default for a spec that places furniture (and off for a 3D-FRONT replay,
# whose dataset poses must stay exact); force it either way with:
PANOWORLD_SETTLE=0 bash scripts/blender_agent_run.sh ...   # off
PANOWORLD_SETTLE=1 bash scripts/blender_agent_run.sh ...   # on (also for front3d)

# per-spec parameters (spec.py ignores unknown top-level keys, so no schema change)
"settle": {"margin_m": 0.002, "drop_m": 0.02, "frames": 60, "substeps": 20,
           "lift_tolerance_m": 0.02, "base_band_m": 0.015, "separation": true}

# headless self-check: placed boxes (floating, sunk, overlapping, stacked, on a
# rug, in a wall, instanced) must end up on the floor, none sunk
blender --background --factory-startup --python blender_agent/settle.py -- --selfcheck
python tests/test_settle.py
```

Environment overrides: `PANOWORLD_SETTLE` (`1`/`0`/`auto`),
`PANOWORLD_SETTLE_<PARAM>` (`MARGIN`, `DROP`, `FRAMES`, `SUBSTEPS`,
`SOLVER_ITERATIONS`, `LIFT_TOLERANCE`, `BASE_BAND`, ...), same names as the spec
block. The build writes `build/settle_report.json` with the per-object numbers
and the list of separations; a settlement that leaves a resting piece floating or
sunk is reported in `report.checks.violations`.

## Lighting model

* World: Nishita sky (azimuth is converted: Blender's `sun_rotation` is measured
  clockwise from +Y) + a sun lamp in the same direction.
* Each window gets an **area light just inside the reveal** aimed into the room:
  it stands in for the skylight and, being a large soft source, is what makes a
  128-sample render clean.  It is *not* visible to camera rays (it is lighting
  gear, and the depth validation raycasts exported geometry only).  A 3D-FRONT
  replay derives these from the dataset's own `Window` meshes
  (`front3d.window_lights`), which is what lights a dataset room: its shell has
  no `windows` entries for the spec-driven portal code to use.
* Lamps (pendant + table) carry a point light; shades have a slight emission.
* The enclosing **sky dome** is what a camera sees through a window and what
  guarantees no ray escapes to the world background (that would leave invalid
  depth pixels).  It is invisible to shadow/diffuse rays, so the room is lit by
  the world sky alone; its emission is deliberately below the world's so the
  window does not blow out under AgX.

## Known constraints on this host

* The container shares an 8 GiB cgroup with other tenants (~5.5 GiB in use), so
  a render must stay under ~2.5 GB.  Consequences baked into the code:
  1k furniture textures, `--no-denoise` during the render with OIDN in a
  separate small process, asset libraries built in their own process, 16 render
  threads, box pixel filter 1.0, a single 2048×1024 render per camera process.
  A 2048×1024 frame peaks around 1.9–2.3 GB (film buffers ≈ 0.7 GB of that).
* trimesh's `RayMeshIntersector` needs the optional `rtree` package, which is
  not installed; the validator ships its own exact caster (median-split BVH for
  the room + analytic sphere/kNN for the dome, float32) and cross-checks it
  against a float64 brute force on a random ray sample.  If `rtree` is present
  it is used as a third opinion.
* Depth is quantised at 1 mm (`place_depth_scale.txt` = 1000), so a median
  agreement of ~0.25 mm against the mesh is the quantisation floor.
* A dataset house is much larger than the demo shell (the real sample scene is
  40 × 6 m with 478 shell meshes), so `wall_thickness` in a bridged spec is a
  nominal 0.12 m: the dataset's walls are zero-thickness planes and the value is
  only used for the top-down drawing.

## P0 → P1 changes worth knowing

* Depth now matches the *furnished* scene (`furnished.ply`), not just the shell.
* `shell.ply` still holds the bare shell for reference; the manifest's
  `visible_geometry.ply` names the file the validator must use.
* The sky dome, window frames/sills, skirting and glass are part of the exported
  geometry, so a window pixel's depth is the glass surface, not the sky.

## P1 → P3 changes worth knowing

* `spec.py` gained the `front3d:` furniture scheme, `scale_xyz` and the
  `front3d` block — all optional, so P0/P1 specs are untouched.
* `run_build.py` accepts `--front3d-scene` / `--front3d-model-root` (and
  `--spec` with a bridged spec); in that mode the shell comes from the dataset,
  the details layer and Poly Haven scene materials are skipped, and furniture is
  placed with the dataset pose instead of the `mount` rules.
* Transmissive glass on dataset `Window` meshes is *not* a problem for the depth
  validation: the Position pass records the pane, so the render stays consistent
  with the exported PLY (verified in the replay test).
