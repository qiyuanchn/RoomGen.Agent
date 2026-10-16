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

## Assets and licensing

* Poly Haven models/textures — **CC0**, fetched on demand via their API.
* MIDI-3D furniture glbs (processed 3D-FRONT/3D-FUTURE) — **CC-BY-NC-4.0**:
  fine for research/evaluation, **not for commercial use**. Swap in the
  official 3D-FRONT + 3D-FUTURE set (email application to the dataset
  authors) for production; `blender_agent/front3d.py` already replays the
  official scene JSON format.
* Code in this repository: Apache-2.0.

## Tests

```bash
python tests/test_front3d_math.py                       # transform-order units
bash  tests/test_front3d_replay.sh                      # replay end-to-end
python tests/test_layout_gen.py                         # layout generator units
bash  scripts/blender_agent_run.sh configs/blender_agent_p0_demo.json outputs/p0
```

Numeric acceptance on the shipped demo: depth-vs-mesh median error 0.25 mm
(16-bit quantisation floor), 100 % valid pixels, ERP orientation within 0.1 px.
