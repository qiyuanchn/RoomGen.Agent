# Demo outputs

Rendered scenes produced by the pipeline, kept in-repo as quality evidence.
Each scene ships its `spec.json` (the generator's contract), a `topdown.png`
floorplan preview, panorama renders per viewpoint, and the numeric
`validation_report.json` (per-pixel depth-vs-geometry raycast check);
`gen_assets_trellis/` additionally ships the placement `checks_report.json`
and the `scene_spec.assets.json` provenance of its generated furniture.

| scene | what it shows | resolution |
| --- | --- | --- |
| `p1_furnished_home/` | hand-authored two-room home: Poly Haven furniture, PBR materials, skirting/casings, door-to-bedroom connectivity | 2048x1024 |
| `generated_seed0/` | **autonomous** 5-room house from `layout_gen.py --seed 0` (bedroom/living/bedroom/study/kitchen, 24 furniture items), strict-pass validation at full resolution | 2048x1024 |
| `generated_seed51/` | autonomous house drawing from the 1050-asset MIDI-3D + Poly Haven library (every house picks different furniture) | 1024x512 |
| `gen_assets_trellis/` | the `gen:` furniture demo: the shipped P1 shell dressed with locally generated assets — TRELLIS text-to-3D sofa, bed, chairs, coffee table and sideboard, no procedural placeholder left | 1024x512 |

Validation highlights (see the JSONs for the full matrix):

* `generated_seed0`: all 10 checks PASS at 2048x1024 — depth median error
  0.25 mm (16-bit quantisation floor) across 5 cameras, 100 % valid pixels.
* `p1_furnished_home` / `generated_seed51` / `gen_assets_trellis`: PASS except
  the `canonical_resolution` gate (these were rendered at their listed size).
* `gen_assets_trellis`: depth median error 0.25 mm and 100 % valid pixels on
  both cameras, doorway width/height and visibility PASS; every `gen:` asset
  in `scene_spec.assets.json` reports `backend: trellis` (the five glbs come
  from `microsoft/TRELLIS-text-xlarge` on an A100). The placement check still
  flags the known 0.24 m bed-to-camera distance inherited from the shipped P1
  shell — see the "Generated assets" section of the top-level README.

全景图是 ERP 等距柱状投影，可用任意全景查看器打开；本仓库对应的
viewer 与数据格式见 README 的 Output Format 一节。

Panoramas are equirectangular (ERP); open with any panorama viewer. The
output format matches the PanoWorld `viewpoints/` convention documented in
the README.
