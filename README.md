# RoomGen.Agent

English | [简体中文](README_zh.md)

Autonomous Blender agent for realistic, multi-room connected indoor scenes.

A seed (or a 3D-FRONT house JSON) becomes a validated, rendered, metric-exact
scene package — panoramas, Euclidean depth, camera poses and adjacency — with
no 2D-generation or reconstruction stage anywhere in the chain.

```bash
# one scene: generate -> check -> build -> render -> export -> validate
bash scripts/layout_gen_run.sh 7 outputs/demo_seed7

# continuous generation: N houses, each a self-contained directory
PANOWORLD_BATCH_RENDER=1 bash scripts/batch_generate.sh 0 100 outputs/stream

# high-quality output
PANOWORLD_RES=2048x1024 PANOWORLD_SAMPLES=128 \
  bash scripts/blender_agent_run.sh <spec.json> <outdir>
```

Rendered examples with validation reports live in [`demo/`](demo/README.md).

## Why build the scene in 3D first

Classic pipelines generate panoramas with diffusion models and then
reconstruct 3D from them; every reconstruction error (depth drift, pose
drift, unobserved regions) then needs compensating machinery. Here the scene
is constructed in Blender and everything else is rendered from it:

| artefact | source | accuracy |
| --- | --- | --- |
| equirect panorama | Cycles path tracing | no generative artefacts |
| metric depth | Position pass -> ray length | mm-quantised, vs mesh median 0.25 mm |
| camera pose | camera matrix | exact |
| instance/semantic masks | Cryptomatte | pixel-exact |
| multi-room connectivity | boolean door openings | by construction |

## Architecture

```
seed / brief / 3D-FRONT JSON
        │
        ▼
layout_gen.py ──► scene_spec.json ◄── front3d.py (dataset replay)
        │              │
        ▼              │
   checks.py           │   collision / clearance / circulation BFS
        │              ▼
        └──► run_build.py (headless Blender)
                shell.py    floor/walls/ceiling + boolean openings
                stage.py    PBR materials, sun+sky, window portals
                assets.py   Poly Haven / MIDI-3D model libraries
                details.py  skirting, casings, bevels, builtin props
                settle.py   rigid-body drop-to-rest, baked into the poses
                panorama.py equirect Cycles + Position-pass depth
                     │
                     ▼
        export_scene.py  viewpoints/<v>/{panoImage_2048.png,
        place_depth.png, place_depth_scale.txt, extrinsics.txt} + map.json
                     │
                     ▼
        tools/validate_blender_agent_p0.py  per-pixel raycast validation
                     │
                     ▼
        blender_agent/review.py  empty rooms, missing fixtures, floating
        furniture, cameras in walls, blown panoramas -> review_report.json
        (+ spec tweaks it can apply itself)   tools/diversity_report.py
```

`scene_spec.json` is the single contract: the planner only emits JSON, the
builder is the only code that touches Blender, and validators close the loop.

## Layout generation (seed → spec)

`blender_agent/layout_gen.py --seed N --out spec.json` turns a seed into a
furnished spec that passes `checks.py` and the render validator: 2–7 rooms,
arranged by a **topology family** — `row` (the default railroad flat), `block`
(the same chain wrapped into two rows with a jog at the corner), `compact_2x2`,
`l_shape`, `central_living` (a radiating plan), `corridor_single` /
`corridor_double`, `loop` (a circulation ring), `u_shape` (two wings over a
band, the notch open) and `terraced` (three bands that step sideways into a
staircase), or `auto` to let the seed draw the room count and the family —
doors from a semantic spanning tree over the shared walls
(kitchen next to dining, bathroom private, bedroom never a public way through),
windows on exterior spans allocated by room function, furniture from the asset
index **plus the functional `builtin:` fixtures** (kitchen counter run with
stove and sink, fridge, bathroom set, desk / chair / bookshelf, dining set —
each with the clear space its use needs), and **cameras sampled on the room's
walkable grid** — at least one per room, scored by what a stand sees
(share of the walkable floor in unobstructed view, furniture in sight, wall and
furniture clearance, distance to the doorways, and whether a doorway is on a
clear sight line).  Small flats get a second living-room camera at least 1.2 m
away whose bearing differs by ≥ 45°; stands stay ≥ 0.6 m off the furniture.
See `blender_agent/README.md` for the family table and the semantics rules.

No two bays are alike, and a plan is rarely a plain rectangle.  Each room draws
the size and the aspect its function wants (`ROOM_SIZE_PREFERENCE`,
`ROOM_ASPECT`: the living room takes the upper part of its area band, a bedroom
the middle, a bathroom the low end) and the family grid is then *fitted* to
those draws, so neighbouring bays of one plan differ, and every room of a flat
differs from the same room in the next seed.  Whole row bands may slide sideways
(the 错动 of `_band_offsets`), which turns an L into a jogged wing and the
terraced family into a staircase; the shift is capped by the shared wall a door
needs, so a shifted plan stays connected.  `auto` draws the arrangement first,
weighted per family (`AUTO_FAMILY_WEIGHT`) and sticky for the seed for the first
attempts (`AUTO_STICKY_ATTEMPTS`), so a sweep is not taken over by whichever
family fails least often — the benchmark read 66/100 `row` before that, with
`central_living` at 0/100.

Nothing pins doors or cameras to a line any more.  The validator used to locate
its door sample column with `atan2(dy, dx)`, which under the `R0` convention is
the *mirror* of the ERP column (a target to the north-east was sampled to the
south-east), so the generator had to keep every door and camera on one common Y
for the check to land on the real sight line.  The projection is now derived
from the camera matrix (`world_to_erp` in `tools/validate_blender_agent_p0.py`,
and the pure-Python twin in `layout_gen.py`), and the door acceptance is local
instead: every opening is measured with horizontal rays from both sides of its
wall — really open, wall really there beside it, width and height against the
spec — while how many doorways a camera sees through is an informational
coverage number that never fails a scene.

`--floors 2` turns the generator into a **duplex**: two storeys on one
footprint, each a row of 2–4 rooms, joined by one switchback stair through a
slab opening.  The spec carries the multi-floor block (`floors`,
`room.floor_id`, `connectors`, `slab_openings`, world-z furniture/cameras), the
stair shaft and its walk-off strip are reserved *before* any furniture is
placed, and `checks.py` validates per storey: the same-floor door graph plus
the stair link, stair geometry, sloped-headroom and a clear stair mouth.

```bash
bash scripts/layout_gen_run.sh 7 outputs/layout_gen/seed7   # spec -> checks -> render -> validate
bash scripts/batch_generate.sh 0 20 outputs/layout_gen/sweep20
```

### Batch grading, exit codes and resume

Every scene summary and `batch_summary.json` grade four levels instead of one
pass count:

| level | means |
|---|---|
| `layout_ok` | spec written and `checks.py` passed |
| `render_ok` | build + render + export produced the complete viewpoint set |
| `validation_ok` | `validation_report.json` passed at the requested resolution (below the canonical 2048x1024 only `canonical_resolution` may fail) |
| `review_ok` | `PANOWORLD_REVIEW=1`: `blender_agent/review.py` found no *failing* issue (see below) |
| `bundle_ok` | the run produced what it was asked for: `layout_ok` for a layout-only sweep, all three levels when `PANOWORLD_BATCH_RENDER=1`, plus `review_ok` when the review was requested |

A render that was requested but did not complete is `pending_retry` (out of
memory) or `failed` — never a pass — and `failure_reasons` / `failed_checks`
carry the distribution.  `layout_gen_run.sh` exits with the failing stage
(`0` ok, `2` layout, `3` checks, `4` render, `5` validation, `6` review) and
stamps a `run_id` into `run_summary.json` so a batch can refuse a stale report;
a memory failure is retried once at a reduced profile (half resolution,
`min(samples, 16)`; override with `PANOWORLD_RETRY_RES` /
`PANOWORLD_RETRY_SAMPLES`).  A scene whose retry *succeeded* at that reduced
profile is a `bundle_ok` but a **degraded** one: the row's `render_quality`
carries the requested and the effective profile, `batch_summary.json` counts
them (`degraded`, `degraded_seeds`) and the table prints
`[degraded 512x256, 16 samples at the requested 1024x512, 24 samples]`, so a
downgraded render can never be read as a completion at the requested quality.

`batch_generate.sh` exits 0 only when every scene reached `bundle_ok`, and an
interrupted run can be finished without repeating a scene that is provably the
same request:

```bash
PANOWORLD_BATCH_RENDER=1 bash scripts/batch_generate.sh 0 100 outputs/stream
PANOWORLD_BATCH_RENDER=1 PANOWORLD_BATCH_RESUME=1 \
  bash scripts/batch_generate.sh 0 100 outputs/stream     # skip provably current scenes
PANOWORLD_BATCH_SUMMARY_ONLY=1 \
  bash scripts/batch_generate.sh 0 100 outputs/stream     # rebuild the table only
PANOWORLD_BATCH_RENDER=1 PANOWORLD_REVIEW=1 \
  bash scripts/batch_generate.sh 0 20 outputs/stream      # + the visual review
```

Resume is bound to a *request manifest*, not to a summary that merely says
`bundle_ok`.  Every scene the batch runs gets `<scene_dir>/resume_manifest.json`
holding

* the request: seed, resolution, samples, floors, shape, rooms, review mode,
  and whether the render was requested;
* the sha256 of every source that decides the spec and the artifacts
  (`layout_gen.py`, `spec.py`, `checks.py`, the build/render/export modules,
  the three scripts, the validator, the layout config), the asset registry the
  generator sizes furniture from, and the git commit for provenance;
* the normalized spec hash plus the room/floor/camera counts it describes;
* the requested and the effective render profile (they differ after a
  memory retry) and the artifact check result.

`PANOWORLD_BATCH_RESUME=1` keeps a scene only when the manifest parses, the
current request hashes to the recorded one, `spec.json` still hashes to the
recorded value, `run_summary.json` is a complete bundle of the requested mode,
and every camera of the spec still has a decodable `panoImage_2048.png` at the
effective resolution, a `uint16` `place_depth.png` of the same size, a positive
`place_depth_scale.txt`, a rigid `extrinsics.txt` plus `map.json` and
`topdown.png`.  Anything else is re-run and the reason is printed
(`(rerun: request changed: render.samples: 4 -> 128; ...)`).  A scene without a
manifest — anything an older batch produced — is never reused, because its
request cannot be proven to match: re-point a pre-manifest output directory
once, or rebuild only the table with `PANOWORLD_BATCH_SUMMARY_ONLY=1`.  The
layout request is explicit, too: `PANOWORLD_BATCH_RENDER` /
`PANOWORLD_RES` / `PANOWORLD_SAMPLES` / `PANOWORLD_ROOMS` /
`PANOWORLD_LAYOUT_FLOORS` / `PANOWORLD_LAYOUT_SHAPE` are resolved (and
exported) by the batch, so the manifest always describes the profile the child
runs really used.

### Visual review and the repair loop

The numeric validators answer *geometric* questions, and a scene can pass all
of them while still being a bad sample: an empty kitchen, a sofa duplicated on
top of itself, furniture that settled floating, a camera pressed into a wall,
a window blown out into a white blob.  `blender_agent/review.py` reads the
package (`spec.json`, `build/build_manifest.json`, `build/settle_report.json`,
the exported panoramas and depth maps) and computes those without a model, a
GPU or Blender — a few hundred milliseconds per scene:

| check | reads | catches |
|---|---|---|
| `rooms_furnished` | spec | a habitable room left empty (a lamp and a plant are not furniture) |
| `functional_coverage` | spec | kitchen without counter/stove/sink, bathroom without a toilet, bedroom without a bed, living room without seating |
| `no_duplicate_furniture` | spec | the same asset placed twice on one spot; four identical repeats in a room |
| `camera_clearance` | spec | a camera in a wall, on the furniture, or with no stand |
| `furniture_grounding` | manifest + settle | furniture floating, sunk or interpenetrating after the settle pass |
| `build_matches_spec` | manifest | a stale build: the render was made from a different spec |
| `exposure` | panoramas + depth | clipped highlight fractions and blobs, very dark viewpoints |
| `camera_framing` | spec | *advisory*: the stand faces away from the room's furniture |
| `camera_coverage` | spec | a room (other than a corridor) without a viewpoint |

Issues are split into **fail** (the scene is unusable) and **warn** (quality
headroom), each with a machine-readable repair hint; `review_report.json` holds
the per-check details, the numbers behind them and the hints.  External
reviewers plug in through `REVIEW_CMD`: it is split with `shlex.split`, gets
the panorama paths appended, and must print one JSON object
(`{"pass": ..., "issues": [...]}`); an explicit reject always fails the scene,
and a reviewer that crashes or prints junk is recorded as
`external.status: "error"` with `retry: true` — that fails too unless the run
opts out with `--allow-external-error`.  Every check carries a four-state
result (`ok` / `issues` / `not_run` / `error`) plus a `required` flag: a
missing required input is `not_run`, never a pass (`--spec-only` narrows the
required scopes to the spec).

```bash
python blender_agent/review.py outputs/stream/scene_0007            # exit 0 pass, 1 fail
python blender_agent/review.py outputs/stream/scene_0007 --strict   # warnings fail too
python blender_agent/review.py outputs/stream/scene_0007 --apply-repair   # patch spec.json
python blender_agent/review.py outputs/stream/scene_0007 --fix-exposure   # redo the colours
REVIEW_CMD="mymodel --json" python blender_agent/review.py <outdir>  # pluggable reviewer
```

A spec tweak (`--apply-repair` above) only helps the *next* render, so
`--fix-exposure` re-does the colour management of the render already on disk
instead: OpenImageDenoise plus the display transform of the saved linear EXRs,
at the exposure those EXRs call for -- no modelling, no ray tracing, minutes
instead of a rebuild.  The previous images stay in `build/pano_pre_exposure/`
and `review_exposure.json` records what changed.  Each camera is rewritten with
the display settings its panorama was *delivered* with, read per camera from
the denoise pass' record in `build/build_manifest.json`
(`render.post_denoise.cameras[]`), then from the render block
(`render.view_transform`/`render.exposure`) for a camera the denoise pass left
to the render, and only for a legacy product that recorded neither from its
actual output (`Standard`/0): an `AgX` panorama is never silently rewritten as
`Standard` -- without Blender the re-do refuses instead
(`unsupported_transform`).  `--exposure N` pins the stop value instead of
estimating one (`--exposure 0` redoes the colours at the exposure the images
already have, and the pixels come back unchanged).

The closed loop is wired into the shell entry points: with
`PANOWORLD_REVIEW=1` a spec-only review runs *before* the render (cheap, it
catches an empty room before a render is paid for) and a full review after it.
When a review fails and `PANOWORLD_REVIEW_REPAIR=1`, `review.py --apply-repair`
turns its hints into spec tweaks — furnish an empty room from the asset index
or the `builtin:` fixtures, move a camera stand to the best free spot in its
room — **validating every candidate with `checks.py` before writing the spec**,
then the run re-checks, re-renders and re-reviews once.  The original spec is
kept as `spec.pre_review.json` and the actions are recorded in
`review_repair.json`.

### Diversity: is this a new house or the same corridor?

`tools/diversity_report.py` measures what a batch actually covers: room count
and type histogram, the **attributed room graph** (one node per room instance
carrying its function and storey, one edge per door / passage / stair / ramp
connection, its `E-V+C` cycle rank and a canonical fingerprint that ignores
names, list order, furniture and size), union footprint and per-storey floor
areas, functional coverage, furniture sets per room type, camera placement
(3D / same-floor / navigation gaps) and lighting.  Topology repeats are
grouped by that canonical fingerprint, so "the same railroad flat with other
furniture" is one topology, while a chain, a star and a ring of equal rooms
are three.

```bash
python tools/diversity_report.py specs outputs/stream/scene_*/spec.json
python tools/diversity_report.py seeds --seeds 0-99 --shape auto --out /tmp/div.json
```

It also carries the batch benchmark: a 100-seed layout sweep, the diversity
statistics over every seed, then a **stratified sample** (at least one scene
per topology family, then seeds that add a new structure hash) rendered at
1024x512 with the review enabled, collated into one report:

```bash
python tools/diversity_report.py benchmark --seeds 0-99 --sample 10 \
  --root outputs/benchmark100 --shape auto --resolution 1024x512 --samples 24 --repair
# -> outputs/benchmark100/{layout,sample}/, diversity.json, report.json
```

`report.json` carries the graded success rates (layout / render / validation /
review / bundle), wall-clock seconds per stage and per scene, the diversity
summary with its duplicate rates, the sample and why each seed was drawn, and
the failure and review-issue distributions.  It refuses to start renders while
another `blender` process is running (use `--force` to override) and reuses
completed scenes with `--resume`.

## Two-storey buildings (multi-floor contract)

The first storey contract is the one described above and keeps working
unchanged (a spec without `floors` is schema_version 1).  A building spec adds
`schema_version: 2` and the storeys:

```jsonc
"schema_version": 2,
"floors": [{"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15},
           {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15}],
"rooms":  [{"id": "hall", "floor_id": "f0", "polygon": [[0,0],[3,0],[3,6],[0,6]]},
           {"id": "landing", "floor_id": "f1", "polygon": [[0,0],[3,0],[3,6],[0,6]]},
           ...],                                    // optional per-room "height"
"connectors": [
  {"kind": "door",  "from_room": "hall", "to_room": "living", "from_floor": "f0",
   "to_floor": "f0", "center": [3.0, 5.5], "width": 0.9, "height": 2.05},
  {"kind": "stair", "from_room": "hall", "to_room": "landing", "from_floor": "f0",
   "to_floor": "f1", "width": 0.85, "run": 2.6, "rise": 2.85, "steps": 16,
   "landing_m": 1.0, "center": [1.5, 3.0], "direction_deg": 90}],
"slab_openings": [{"outline": [[0.65,1.2],[2.35,1.2],[2.35,4.95],[0.65,4.95]],
                   "floors": ["f1"]}]
```

* A floor's `elevation` is its **walking surface**, `height` its net height and
  `slab_thickness` the slab **below** it (`[elevation - slab_thickness,
  elevation]`).  One storey's ceiling is the next storey's slab — the builder
  never emits both, and a spec whose upper slab would reach below the lower
  ceiling is rejected before anything is built.
* Walls merge **within one storey only**; door and window cutters are applied
  per storey (a second-storey door cannot notch a first-storey wall even though
  both share their XY), and window `sill`/`height` stay relative to the room
  floor.
* `connectors` carry every connection (`door`/`open_passage` on one floor,
  `stair`/`ramp` between two).  `slab_opening` punches its outline through the
  floor slab of each listed storey — the stairwell.  Legacy `openings` still
  work and are folded into the same connection list.
* Stairs are **built geometry**: a straight run (`landing_m: 0`) or a switchback
  with a half landing, `steps` risers of `rise / steps`, treads of
  `run / (steps/2)` per flight, solid blocks from the lower floor up.
* Furniture and camera `z` are **world** heights: the bottom of a piece standing
  on a room's floor is that floor's `elevation`, the settle pass drops it onto
  the same plane, and everything written out (manifest, extrinsics, settled
  poses) stays in world coordinates.

`checks.py` verifies the 2.5D building: one walkable grid per storey (slab
openings are holes, and a stair footprint is never standable floor), with
stair/ramp **walk tracks** carrying their own heights — riser / tread / width
/ pitch / landing bands, monotone climb, ≥ 2.0 m body clearance along the
track, and the track joining the floor grids only at its two ends.  Cameras
must stand on real floor with a support surface beneath and the eye height
above it (`cameras_stand`); the fly-through opening in the slab above, a clear
stair mouth, and — because the walk bands are per storey — furniture stacked
in XY on two floors is never a false collision.  Ceilings are computed per
area: a storey slab replaces only the ceiling it actually covers, so a partial
upper floor leaves the rest of the room below roofed (`ceiling_regions`).  `settle.py` drops each piece onto its own storey's floor,
`export_scene.py` writes `topdown_<floor>.png` per storey (plus a stacked
`topdown.png` and a `connectors.json`), and `map.json` links two viewpoints
when their rooms share a door **or a stair** — the cross-floor adjacency.
`tools/validate_blender_agent_p0.py` measures every door inside its own storey,
probes each step top and each slab opening against the shell, and checks that
the upper slab and the lower ceiling are exactly one slab sandwich.

```bash
# hand-written two-storey demo (4 rooms per storey, one switchback stair)
python blender_agent/checks.py tests/fixtures/two_floor_demo.json
PANOWORLD_RES=1024x512 PANOWORLD_SAMPLES=24 \
  bash scripts/blender_agent_run.sh tests/fixtures/two_floor_demo.json outputs/two_floor_demo
```

Known limits of this first version: `details.py` (skirting, casings, window
glass) is not storey-aware, so the details layer is skipped on multi-floor
specs; `stage.build_window_portals` still places its area lights at the opening
height without the storey elevation (leave `lighting.window_portal_w` at 0 on a
multi-floor spec, lamps still work); the ceiling of a room under another storey
shows the upper floor's material where the slab covers it; and only straight /
switchback stairs and solid ramps are implemented — no curved flights, no
mezzanines, and a ramp that climbs a whole storey needs a long run
(the slope band is a loose 30 %).

## Layout

```
blender_agent/    spec, builder, render, export, layout generation, checks, review
scripts/          one-command entry points (incl. fetch_assets.sh)
tools/            asset indexing/merging, numeric validation, diversity benchmark
configs/          demo specs (hand-authored and generated)
tests/            unit + end-to-end tests (incl. 3D-FRONT replay fixture)
demo/             rendered scenes + validation reports
third_party/      local toolchain and asset data (not tracked)
  blender-*/      Blender 4.5 LTS binary
  assets/         model/texture caches + registries, built by fetch_assets.sh
```

## Setup

```bash
# 1. Blender 4.5 LTS (headless) into third_party/
mkdir -p third_party && cd third_party
curl -L -o blender.tar.xz \
  https://mirrors.tuna.tsinghua.edu.cn/blender/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz
tar xf blender.tar.xz && cd ..

# 2. asset library (downloaded + indexed, kept out of git)
bash scripts/fetch_assets.sh                 # Poly Haven + MIDI-3D (~13.3 GB)
bash scripts/fetch_assets.sh --skip-midi3d   # fast path (~200 MB)

# 3. host python with numpy/PIL (validation + indexing)
```

`third_party/assets/` is intentionally **not** tracked: it is bulk data the
fetch script rebuilds from source (registries included). Override tool paths
with `PANOWORLD_BLENDER` / `PANOWORLD_HOSTPY` if needed.

## Generated assets (`gen:`)

Where Poly Haven + MIDI-3D leave a gap (sofas, beds and chairs run out first),
generated furniture fills it: `tools/gen_assets_trellis.py` writes glbs into
`third_party/assets/generated/` and registers them through the *same* index +
library machinery as `ph:`/`midi3d:` (`blender_agent/assets.py`), so a spec only
says `gen:<id>`:

```jsonc
"furniture": [{"asset": "gen:sofa_01", "room": "living", "pos": [3.7, 3.6], "rot_z_deg": 0}]
```

```bash
# one command: fill the cache -> emit a demo spec -> standard 6-stage pipeline
bash scripts/gen_assets_trellis.sh outputs/gen_assets
bash scripts/gen_assets_trellis.sh outputs/gen_assets proc      # force the procedural backend

# stage by stage
python tools/gen_assets_trellis.py generate --backend proc      # trimesh fallback, no GPU
python tools/gen_assets_trellis.py list
python tools/gen_assets_trellis.py emit-spec --out outputs/gen_assets/scene_spec.json
```

Two backends (`--backend auto`, the default, picks trellis when a usable GPU
container is found, otherwise proc):

| backend | what it produces | needs |
|---|---|---|
| `proc` | trimesh box assemblies (sofa/armchair/chair/stool/bed/table/cabinet/lamp) | host python + trimesh; no GPU, no network |
| `trellis` | real TRELLIS text-to-3D (`microsoft/TRELLIS-text-xlarge`) | a GPU container + TRELLIS checkout + weights |

This host has no GPU, so the `trellis` backend runs the generator inside a GPU
container with `docker exec`; the container must mount the repo and the TRELLIS
checkout at the **same absolute paths**:

```bash
docker run -d --name roomgen-trellis --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 \
  --user $(id -u):$(id -g) -v /srv/TRELLIS:/srv/TRELLIS:ro \
  -v /srv/conda-envs/trellis:/srv/conda-envs/trellis -v "$PWD:$PWD" -w /srv/TRELLIS \
  <cuda-image> sleep infinity

python tools/gen_assets_trellis.py fetch-weights --container roomgen-trellis
python tools/gen_assets_trellis.py generate --backend trellis \
  --container roomgen-trellis --classes sofa bed chair --variants 2
```

Override the paths with `PANOWORLD_TRELLIS_REPO` / `_PYTHON` / `_CONTAINER` /
`_HF` (weights default to `third_party/assets/trellis_hf/`, downloaded through
`HF_ENDPOINT`, `https://hf-mirror.com` by default).  Every generated asset is
normalised exactly like a Poly Haven model (uniform scale onto the category's
target size, horizontal principal-axis alignment, bounding-box bottom centre at
the origin, front facing -Y), so specs need no extra parameters; `emit-spec`
also writes each `gen:` asset's provenance (backend / prompt / seed / measured
size) to `<spec>.assets.json`, next to a `checks_report.json` from
`blender_agent/checks.py`.

Known limits: the facing of a text-to-3D asset is a heuristic (TRELLIS has no
canonical front) and can be turned with `CATEGORIES[...]["yaw_deg"]` or `--yaw`;
in the demo shell the bed sits 0.25 m from the bedroom camera (the shipped P1
spec is at 0.30 m -- the same geometric constraint).

## Viewing scenes in the browser

Two viewers answer two different questions:

| viewer | shows | start it with |
|---|---|---|
| panorama tour (PanoWorld, port 8003) | the *rendered* ERP panoramas - Cycles lighting, sky, denoising - with hotspot jumps between viewpoints | `panoworld_viewer_server.py --viewpoints <outdir>/scene/viewpoints` |
| 3D scene viewer (this repo, port 8004) | the *geometry* itself - shell, furniture, PBR materials at 1:1 scale, from any angle | `bash scripts/scene3d_view.sh <outdir>` |

The 3D viewer is a static page under `tools/scene3d_viewer/` (three.js r180
vendored locally, no build step, no CDN); it loads the `scene.glb` written by
`run_build.py --export-glb`:

```bash
# full pipeline + GLB, or the geometry-only fast path used for the demo scenes
PANOWORLD_EXPORT_GLB=1 bash scripts/blender_agent_run.sh configs/blender_agent_p1_furnished.json outputs/p1
third_party/blender-4.5.14-linux-x64/blender --background --factory-startup \
  --python blender_agent/run_build.py -- --spec configs/blender_agent_p1_furnished.json \
  --out outputs/blender_agent/p1_settled/build --skip-render --export-glb

# serve one scene, then open http://<host>:8004/
bash scripts/scene3d_view.sh outputs/blender_agent/p1_settled   # Poly Haven furnished home
bash scripts/scene3d_view.sh outputs/gen_assets                 # TRELLIS-generated furniture

# re-read the GLB without Blender (meshes, materials, bounds, cameras-inside check)
/opt/conda/envs/diffaero/bin/python tools/check_scene_glb.py outputs/blender_agent/p1_settled
```

What the page offers: Orbit (drag / wheel / right-drag), Fly and Walk (WASD +
mouse look under pointer lock, Shift sprints; Walk is pinned to the 1.5 m eye
height, Fly adds Space/C for up/down), a viewpoint picker that jumps to every
panorama camera pose and room, a load-progress bar, and hemisphere + directional
+ IBL lighting over the GLB's own materials.

What the GLB holds: what the panoramas see, minus the render-only sky dome and
minus cameras/lights (the page brings its own). For the P1 demo home that is 53
meshes in 61 nodes, 743,627 triangles, 27 materials, 61 textures/images, 52.5 MB,
in glTF's +Y-up metres. `build/scene3d_manifest.json` carries the camera poses
and room boxes already converted to glTF axes, so the page never guesses, and
`tools/check_scene_glb.py` re-checks those numbers against the geometry (bounds
within millimetres, every camera inside the room).

## Assets and licensing

* Poly Haven models/textures — **CC0**, fetched on demand via their API.
* MIDI-3D furniture glbs (processed 3D-FRONT/3D-FUTURE) — **CC-BY-NC-4.0**:
  fine for research/evaluation, **not for commercial use**. Swap in the
  official 3D-FRONT + 3D-FUTURE set (email application to the dataset
  authors) for production; `blender_agent/front3d.py` already replays the
  official scene JSON format.
* `gen:` assets are produced locally and carry no third-party asset licence:
  the `proc` geometry is code in this repository (Apache-2.0), and the
  `trellis` backend uses the MIT-licensed TRELLIS code and
  `microsoft/TRELLIS-text-xlarge` weights (also MIT).
* Code in this repository: Apache-2.0.

## Tests

```bash
python tests/test_front3d_math.py                       # transform-order units
bash  tests/test_front3d_replay.sh                      # replay end-to-end
python tests/test_layout_gen.py                         # layout generator units
python tests/test_settle.py                             # furniture settling units + box self-check
python tests/test_multifloor.py                         # storey contract, stairs, 2.5D checks, probes
python tests/test_batch_stats.py                        # batch grading, exit codes, retry, resume manifests
python tests/test_review.py                             # review checks, repair, the review hooks
bash  scripts/blender_agent_run.sh configs/blender_agent_p0_demo.json outputs/p0
```

Numeric acceptance on the shipped demo: depth-vs-mesh median error 0.25 mm
(16-bit quantisation floor), 100 % valid pixels, ERP orientation within 0.1 px.
