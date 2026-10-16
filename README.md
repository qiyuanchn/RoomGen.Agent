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
```

`scene_spec.json` is the single contract: the planner only emits JSON, the
builder is the only code that touches Blender, and validators close the loop.

## Layout generation (seed → spec)

`blender_agent/layout_gen.py --seed N --out spec.json` turns a seed into a
furnished spec that passes `checks.py` and the render validator: 2–5 rooms in a
row (one door per shared wall, all on a common door line near the south edge),
windows on exterior spans, furniture placed from the asset index by room-type
templates, and **cameras sampled on the room's walkable grid** — at least one per
room, scored by what a stand sees (share of the walkable floor in unobstructed
view, furniture in sight, wall and furniture clearance, distance to the
doorways).  Small flats get a second living-room camera at least 1.2 m away
whose bearing differs by ≥ 45°.  Stands stay in the band around the door line
and ≥ 0.6 m off the furniture — the depth validator samples the *mirrored* door
column, which only reaches the doorway from there (`blender_agent/layout_gen.py`
explains why).

```bash
bash scripts/layout_gen_run.sh 7 outputs/layout_gen/seed7   # spec -> checks -> render -> validate
bash scripts/batch_generate.sh 0 20 outputs/layout_gen/sweep20
```

## Layout

```
blender_agent/    spec, builder, render, export, layout generation, checks
scripts/          one-command entry points (incl. fetch_assets.sh)
tools/            asset indexing/merging, numeric validation
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
bash  scripts/blender_agent_run.sh configs/blender_agent_p0_demo.json outputs/p0
```

Numeric acceptance on the shipped demo: depth-vs-mesh median error 0.25 mm
(16-bit quantisation floor), 100 % valid pixels, ERP orientation within 0.1 px.
