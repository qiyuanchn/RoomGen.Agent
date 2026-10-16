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
| `review.py` | host: low-cost visual/rule review of a package (`review_report.json`) + the spec repair the review hints imply |
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
| `block` | the same chain wrapped into two rows, an L / block outline with a jog at the corner |
| `compact_2x2` | 4 rooms in a 2x2 block (living with the kitchen beside it, bedroom and bathroom behind) |
| `l_shape` | an L: dining and kitchen together, living below, bathroom off it, bedroom at the end |
| `central_living` | the living room spans the middle band, rooms open off it north and south (radiating) |
| `corridor_single` | a hall strip along one side, rooms over it, living closing the end |
| `corridor_double` | rooms on both sides of a middle hall |
| `loop` | a circulation ring (living -> hall -> kitchen -> dining -> living) with bedroom and bathroom as leaves |
| `u_shape` | a U: two wings over a south band, the notch open to the outside, bathroom as the bedroom's en-suite |
| `terraced` | three bands that step sideways one after the other -- a staircase outline out of plain rectangles |
| `auto` | the seed draws the room count and one of the families (`PANOWORLD_LAYOUT_SHAPE=auto bash scripts/layout_gen_run.sh <seed> <out>`) |

Rooms are rectangles (a family's outline can be an L, a U or a staircase); the
functional programme covers living room, bedroom(s), kitchen, bathroom plus
dining, study, entry, corridor, storage, laundry and balcony.  The room *type*
of a cell follows the drawn size, so a small service cell becomes a bathroom
and never a 3 m2 bedroom.  The sizes themselves are drawn, not fixed: every room
samples an area out of its type's preference slice (`ROOM_SIZE_PREFERENCE`,
which keeps the living > bedroom > bathroom hierarchy) and an aspect out of
`ROOM_ASPECT` (a hall runs long and narrow, a living room stays squarish), and
the family grid is fitted to those draws -- so neighbouring bays of one plan
differ, and whole row bands may slide sideways up to the shared wall a door
needs (`_band_offsets`, the 错动 of the terraced / jogged outlines).

`--shape auto` draws the arrangement first, from `AUTO_FAMILY_WEIGHT` (the
railroad `row` carries the smallest weight, the two-dimensional plans the
largest) and keeps it for the seed's first attempts (`AUTO_STICKY_ATTEMPTS`).
Re-drawing the family on every attempt used to hand the sweep to whichever
family failed least often: the benchmark read 66/100 `row` that way, with
`central_living` at 0/100.

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
python blender_agent/layout_gen.py --seed 3 --floors 2 --out /tmp/duplex.json   # two storeys
python blender_agent/checks.py /tmp/spec.json          # placement + circulation
```

### Two-floor duplexes (`--floors 2`)

`--floors 2` (also `PANOWORLD_LAYOUT_FLOORS=2` for the scripts) turns the
generator into a duplex: **two floors on one footprint**, each a row of 2–4
rooms, joined by one switchback stair through a slab opening.  `--rooms` then
means the total (4–8, split between the floors) and `--shape` must stay
`row`/`auto` — the duplex draws its own arrangement.

* **Floor programmes**: floor 1 is the public floor (living room and kitchen
  always, plus an entrance hall, dining room or bathroom); floor 2 is the
  sleeping floor — an upstairs **hall** (the stair lands there, which is also
  what leaves a bed or sofa room to stand beside the shaft), a bedroom, a
  bathroom, optionally a study or a second bedroom / small living room.  The
  whole home always has at least one bedroom and one bathroom.
* **One stair**: a switchback of 16 risers (2.90 m floor-to-floor at 0.18 m a
  step), 0.95 m wide flights, a 0.85 m half landing, leaning on the south wall
  inside one room of each storey.  The stair shaft and its walk-off strip are
  **reserved before any furniture is placed** (`Placer.reserved`), a pendant
  lamp may hang over the strip but never over the stair itself, and the host
  room's placements must keep the stair mouth connected to its doorways.
* **Spec block** (the multi-floor contract of `spec.py`, schema version 2):
  `floors` with `elevation`/`height`/`slab_thickness`, `room.floor_id`,
  `connectors` (the stair: kind, rooms, floors, width, run, rise, steps,
  landing_m, centre and climbing direction), `slab_openings` (the shaft cut out
  of the upper slab) and world-z furniture/cameras.  Doors and windows stay in
  the single-floor format, floor-local; wall segments merge per floor only.
* **Verification**: every duplex must pass `checks.py`'s per-storey rule set
  (same-floor doors plus the stair link, stair geometry / opening coverage /
  headroom / clear mouth) *and* `layout_gen.check_duplex_spec`, the
  generator-side twin that additionally pins the programme, the world-z rules,
  furniture outside the reserved shaft and reachability from the entrance.

```bash
PANOWORLD_LAYOUT_FLOORS=2 bash scripts/layout_gen_run.sh 3 outputs/layout_gen/duplex3
python blender_agent/layout_gen.py --seed 3 --floors 2 --out /tmp/duplex.json
python blender_agent/checks.py /tmp/duplex.json        # per-storey + stairs, exit 0
```

## Visual review (`review.py`)

`checks.py` and the render validator answer geometric questions; `review.py`
answers the "would a person keep this sample?" ones.  It is host-side and
cheap (no Blender, no model, no GPU; a few hundred ms per scene) and reads what
the pipeline already wrote:

* **spec scope** -- `rooms_furnished` (a habitable room holding only a lamp and
  a plant is *empty*), `functional_coverage` (kitchen without a counter, stove
  or sink; bathroom without a toilet; bedroom without a bed; living room
  without seating; study without a desk or shelving), `no_duplicate_furniture`
  (the same asset twice on one spot, or four identical repeats in a room),
  `camera_clearance` (a stand in a wall, on the furniture, or inside its
  comfort band: `checks.py` allows 0.10 m, the review warns under 0.35 m),
  `camera_framing` (advisory -- a panorama sees the whole sphere, so a stand
  facing the emptiest side of its room is a quality note, not a defect), and
  `camera_coverage` (a room other than a corridor without a viewpoint);
* **build scope** -- `furniture_grounding` reads `furniture[i].settle` from
  `build/build_manifest.json` and flags a piece that settled more than 2 cm up
  (warn) / 5 cm up or 2 cm down (fail) or is still interpenetrating, with
  `builtin:rug`-style floor finish and ceiling/wall mounts exempt;
  `build_matches_spec` fails a build whose manifest carries a *different* spec
  (the stale-build trap after a repair);
* **render scope** -- `exposure` decodes every `panoImage_2048.png`,
  measures the clipped-pixel fraction and the largest clipped blob (35 % / 25 %
  fails, 15 % / 8 % warns -- shipped scenes sit at 0-25 %), and warns on a very
  dark viewpoint; the depth maps add median depth and the near-content share of
  the horizon band as context;
* **external** -- `REVIEW_CMD` (env) is split with `shlex.split`, gets the
  panorama paths appended and must print one JSON object; a broken hook is
  reported as `external.status: "error"` and never fails a scene by itself.

`--strict` promotes warnings to failures.  Every issue carries a `repair` hint,
and `--apply-repair` applies the ones that are safe as **spec tweaks**:

```bash
python blender_agent/review.py outputs/stream/scene_0007             # report + exit code
python blender_agent/review.py outputs/stream/scene_0007 --spec-only # no build/render reads
python blender_agent/review.py outputs/stream/scene_0007 --apply-repair
```

* `add_furniture` -- a fixture (or an index asset from the matching
  `layout_gen` slot: sofa for the living room, bed for the bedroom) placed
  against a wall of the empty room, keeping the clear space its use needs;
* `move_camera` -- the best free stand in the room (furniture in front, off the
  walls, 0.60 m from the furniture, clear of the door zones);
* `reduce_exposure` -- a blown-out panorama lowers `style.exposure` a little.

Each candidate is validated with `checks.check_spec` *before* the spec is
written, so a repair can never publish a layout the placement checks reject;
the previous spec is kept as `spec.pre_review.json`, the actions are recorded
in `review_repair.json`, and `scripts/layout_gen_run.sh` drives the loop
(`PANOWORLD_REVIEW=1`, plus `PANOWORLD_REVIEW_REPAIR=1` for one repair retry,
graded as `review_ok` in the summary and exit code 6 when it fails).

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
  The point-light wattage is `lighting.lamp_w`, **12 W** by default: at the old
  8 W the windowless service rooms (bathroom, hall, storage) rendered visibly
  dark even with their pendant on.
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
