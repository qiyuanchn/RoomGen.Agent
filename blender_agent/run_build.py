"""Blender-side entry point: build the scene and render PanoWorld-style panos.

Usage::

    blender --background --factory-startup --python blender_agent/run_build.py \
        -- --spec configs/blender_agent_p1_furnished.json --out outputs/.../build [--selftest]

Everything after ``--`` is parsed by this script; Blender itself ignores it.
Outputs (inside ``--out``): ``pano_<cam>.png``, ``depth_<cam>.npy``,
``position_<cam>*.exr``, ``raw_<cam>*.exr``, ``shell.ply`` (room shell only),
``furnished.ply`` (everything the cameras can see), ``selftest_report.json``,
``build_manifest.json``, and with ``--export-glb``: ``scene.glb`` +
``scene3d_manifest.json`` for the browser 3D viewer (see ``tools/scene3d_viewer``).

P1 additions (all optional in the spec; a P0 spec keeps working unchanged):
``furniture`` (Poly Haven models and ``builtin:`` props), ``materials`` (PBR
texture sets), ``details`` (skirting / casings / bevels / glass) and
``lighting`` (window portal area lights, lamp point lights).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bpy  # noqa: E402  (must come after the sys.path fix)

from blender_agent import assets, details, export_glb, front3d, materials, panorama, render_selftest, settle, shell  # noqa: E402
from blender_agent import spec as spec_mod, stage  # noqa: E402

DEFAULT_CACHE = os.path.join(REPO_ROOT, "third_party", "assets", "polyhaven")
FURNITURE_COLLECTION = "furniture"
DETAILS_COLLECTION = "details"


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="run_build.py", description="Build a PanoWorld shell scene in Blender")
    parser.add_argument("--spec", help="path to the scene_spec JSON (omit when --front3d-scene is given)")
    parser.add_argument("--front3d-scene", help="3D-FRONT scene JSON to replay (P3)")
    parser.add_argument("--front3d-model-root", help="path to the 3D-FUTURE-model folder tree")
    parser.add_argument("--front3d-texture-root", default=None, help="path to 3D-FRONT-textures")
    parser.add_argument("--front3d-camera-source", default="room", choices=["room", "snapshots"],
                        help="viewpoints: one per room (default) or the dataset's own snapshots")
    parser.add_argument("--front3d-eye-height", type=float, default=front3d.DEFAULT_EYE_HEIGHT)
    parser.add_argument("--front3d-max-cameras", type=int, default=0, help="0 = pano per room")
    parser.add_argument("--front3d-wall-thickness", type=float, default=front3d.DEFAULT_WALL_THICKNESS)
    parser.add_argument("--out", required=True, help="output build directory")
    parser.add_argument("--selftest", action="store_true", help="run the ERP orientation self-test first")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--width", type=int, default=spec_mod.PANO_WIDTH)
    parser.add_argument("--height", type=int, default=spec_mod.PANO_HEIGHT)
    parser.add_argument("--threads", type=int, default=16, help="render thread count (memory bound in containers)")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="Poly Haven asset cache directory")
    parser.add_argument("--rebuild-assets", action="store_true", help="rebuild the .blend asset libraries")
    parser.add_argument("--tex-resolution", default="1k", choices=["1k", "2k", "4k"],
                        help="furniture texture variant to build libraries from (1k saves ~4x texture memory)")
    parser.add_argument("--no-denoise", action="store_true", help="skip in-render OpenImageDenoise; pair with --denoise-only")
    parser.add_argument(
        "--denoise-only",
        action="store_true",
        help="no geometry/render: OpenImageDenoise each camera's raw EXR from the manifest",
    )
    parser.add_argument("--denoise-prefilter", default="ACCURATE", choices=["NONE", "FAST", "ACCURATE"])
    parser.add_argument("--denoise-quality", default="FAST", choices=["HIGH", "BALANCED", "FAST"])
    parser.add_argument("--prepare-assets-only", action="store_true",
                        help="only build the .blend asset libraries, then exit (keeps the render process lean)")
    parser.add_argument("--skip-render", action="store_true", help="build geometry only (fast smoke test)")
    parser.add_argument("--export-glb", action="store_true",
                        help="also write scene.glb + scene3d_manifest.json for the browser 3D viewer")
    parser.add_argument("--export-glb-path", default=None,
                        help="scene.glb destination (default: <out>/scene.glb)")
    parser.add_argument("--export-glb-include-environment", action="store_true",
                        help="keep the render-only sky dome in the GLB (off by default)")
    return parser.parse_args(argv)


def run_denoise_pass(build_dir: str, prefilter: str = "ACCURATE", quality: str = "FAST") -> int:
    """Denoise every camera's raw EXR from the manifest (low-memory Blender run)."""
    manifest_path = os.path.join(build_dir, "build_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    reports = []
    for entry in manifest.get("cameras", []):
        raw = entry.get("raw_exr")
        if not raw:
            print(f"[run_build] camera {entry['camera']}: no raw EXR recorded, skipping denoise")
            continue
        raw_path = os.path.join(build_dir, raw)
        pano_path = os.path.join(build_dir, entry["pano"])
        tick = time.time()
        render_info = manifest.get("render", {})
        report = panorama.denoise_pano_file(
            raw_path, pano_path, prefilter=prefilter, quality=quality,
            view_transform=render_info.get("view_transform", "Standard"),
            exposure=float(render_info.get("exposure", 0.0) or 0.0),
        )
        report["camera"] = entry["camera"]
        report["seconds"] = round(time.time() - tick, 2)
        reports.append(report)
        print(
            f"[run_build] denoised {entry['camera']}: {raw} -> {entry['pano']} "
            f"({report['size'][0]}x{report['size'][1]}, {report['seconds']}s, oidn {prefilter}/{quality})"
        )
    manifest.setdefault("render", {})["post_denoise"] = {
        "denoiser": "OPENIMAGEDENOISE",
        "prefilter": prefilter,
        "quality": quality,
        "cameras": reports,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[run_build] denoise pass done, manifest updated: {manifest_path}")
    return 0 if reports else 1


# --------------------------------------------------------------------------- #
# P1 scene dressing: details, materials, furniture, lights
# --------------------------------------------------------------------------- #
def build_details_layer(spec: spec_mod.SceneSpec, shell_info: dict) -> dict:
    """Skirting, casings, window glass and shell edge bevels.

    On a multi-floor spec the layer is skipped: ``details.py`` builds its trim on
    the ground plane (skirting at ``z = 0``, casings and glass at the opening's
    own height) and is not storey-aware yet, so a second-storey door would get
    its casing on the ground floor.  The shell itself stays storey-aware.
    """
    report = {"skirting": 0, "casings": 0, "glass": 0, "bevels": 0}
    if spec.floors:
        report["skipped"] = "details.py is not storey-aware yet (multi-floor spec)"
        return report
    config = spec.details or {}
    collection = shell.ensure_collection(DETAILS_COLLECTION)
    objects = []
    if config.get("skirting", False):
        pieces = details.build_skirting(spec, collection)
        report["skirting"] = len(pieces)
        objects += pieces
    if config.get("casings", False):
        pieces = details.build_casings(spec, collection)
        report["casings"] = len(pieces)
        objects += pieces
    if objects:
        trim = materials_mod_plain_principled("mat_trim", (0.93, 0.92, 0.90), 0.45)
        for obj in objects:
            obj.data.materials.append(trim)
    glass = []
    if config.get("glass", True) and spec.windows:
        glass = details.build_glass(spec, collection)
        glass_material = details.make_glass_material()
        for obj in glass:
            obj.data.materials.append(glass_material)
        objects += glass
        report["glass"] = len(glass)
    if config.get("bevels", True):
        report["bevels"] = details.bevel_shell(list(shell_info["shell_objects"]) + objects)
    shell_info["details_objects"] = objects
    shell_info["glass_objects"] = glass
    return report


def materials_mod_plain_principled(name: str, color, roughness: float):
    """Flat Principled material (the P0 look, reused for trim/plain surfaces)."""
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        bsdf = material.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Roughness"].default_value = float(roughness)
    return material


def prepare_assets(spec: spec_mod.SceneSpec, cache_dir: str, force: bool = False,
                   resolution: str = "1k") -> dict:
    """Resolve every furniture asset: cache check + library build (wipes the file!).

    ``ph:<id>`` resolves in the Poly Haven cache (``cache_dir``); ``midi3d:<id>``
    and ``gen:<id>`` resolve in the sibling ``third_party/assets/{midi3d,generated}``
    caches, whose index.json files are written in the same shape by
    ``tools/make_merged_assets.py`` and ``tools/gen_assets_trellis.py``.
    """
    caches = {
        "ph": cache_dir,
        "midi3d": assets.sibling_cache(cache_dir, "midi3d"),
        "gen": assets.sibling_cache(cache_dir, "generated"),
    }
    groups = {scheme: set() for scheme in caches}
    for item in spec.furniture:
        scheme, _, asset_id = item.asset.partition(":")
        if scheme in groups:
            groups[scheme].add(asset_id)
    libraries = {}
    for scheme, ids in groups.items():
        if not ids:
            continue
        group_cache = caches[scheme]
        index = assets.load_index(group_cache)
        missing = []
        for asset_id in sorted(ids):
            try:
                kind, entry = assets.asset_entry(index, asset_id)
            except KeyError:
                missing.append(asset_id)
                continue
            if kind != "model":
                missing.append(f"{asset_id} (is a texture)")
            elif resolution not in (entry.get("gltf_variants") or {}):
                missing.append(f"{asset_id} (no {resolution} variant)")
        if missing:
            if scheme == "ph":
                hint = ("fetch them with: python blender_agent/assets.py fetch --resolution "
                        f"{resolution} --assets " + " ".join(f"ph:{m.split(' ')[0]}" for m in sorted(missing)))
            elif scheme == "midi3d":
                hint = "rebuild the midi3d cache index with: python tools/make_merged_assets.py"
            else:
                classes = sorted({m.split(" ", 1)[0].rsplit("_", 1)[0] for m in missing})
                hint = ("generate them with: python tools/gen_assets_trellis.py generate --classes "
                        + " ".join(classes))
            raise SystemExit(f"[run_build] missing {scheme} assets: "
                             + ", ".join(sorted(missing)) + f"\n  {hint}")
        meta = assets.load_asset_meta() if scheme == "ph" else {}
        libraries.update(assets.ensure_libraries(group_cache, sorted(ids), meta, force=force,
                                                 resolution=resolution))
    return libraries


def place_furniture(spec: spec_mod.SceneSpec, libraries: dict, scene_materials: dict,
                    front3d_ctx: dict = None) -> dict:
    """Place models as collection instances plus the generated ``builtin:`` props.

    ``front3d_ctx`` (P3 replay) carries ``model_root`` plus the import cache and
    report dicts; ``front3d:<jid>`` items are placed with the dataset's own
    position / yaw / per-axis scale instead of the Poly Haven mounting rules
    (their ``z`` is the dataset height, not "sit on the floor at 0").
    """
    collection = shell.ensure_collection(FURNITURE_COLLECTION)
    placements = []
    lamp_lights = []
    skipped = []
    for index, item in enumerate(spec.furniture):
        name = item.name or f"item_{index}"
        if item.asset.startswith("front3d:"):
            if front3d_ctx is None:
                raise SystemExit(
                    f"[run_build] '{item.asset}' needs a 'front3d' block in the spec "
                    f"(or --front3d-scene); the model root is unknown otherwise"
                )
            jid = item.asset.split(":", 1)[1]
            model = front3d.import_model(front3d_ctx["model_root"], jid,
                                         cache=front3d_ctx.setdefault("models", {}),
                                         report=front3d_ctx.setdefault("report", {}))
            if model is None:
                # the 3D-FUTURE model set is only partially public; the dataset's
                # own loader skips these too
                skipped.append({"name": name, "jid": jid, "reason": "model folder missing or empty"})
                continue
            matrix = front3d.item_matrix(item.pos, float(item.z or 0.0), item.rot_z_deg,
                                         item.scale_xyz or [item.scale] * 3)
            empty = front3d.place_model(model, f"front3d_{name}", matrix, collection=collection,
                                        properties={"front3d_jid": jid, "front3d_name": name})
            bbox = front3d.model_world_bbox(model, matrix)
            placements.append({
                "name": name, "asset": item.asset, "kind": "front3d", "instance": empty.name,
                "jid": jid, "room": item.room, "pos": [float(v) for v in item.pos],
                "rot_z_deg": item.rot_z_deg, "z": float(item.z or 0.0),
                "scale": float(item.scale), "scale_xyz": [float(v) for v in (item.scale_xyz or [1.0] * 3)],
                "matrix_world": [[float(matrix[i][j]) for j in range(4)] for i in range(4)],
                "model_source": os.path.relpath(model["source"], REPO_ROOT)
                if model["source"].startswith(REPO_ROOT) else model["source"],
                "model_bbox": model["local_bbox"],
                "model_origin_y": round(model["bbox"][0][1], 6) if model["bbox"] else None,
                "model_vertices": model["vertices"], "model_polygons": model["polygons"],
                "bbox_world": bbox,
            })
            continue
        if item.asset.startswith("builtin:"):
            kind = item.asset.split(":", 1)[1]
            floor_z = spec.room_elevation(item.room)
            if kind == "rug":
                size = item.size
                obj = details.make_rug(
                    item.pos, size, item.rot_z_deg,
                    scene_materials.get("rug"), collection, name=f"rug_{name}",
                )
                if scene_materials.get("rug") is not None:
                    obj.data.materials.append(scene_materials["rug"])
                # make_rug builds on the ground plane; lift it to the room's storey
                obj.location.z += floor_z
                placements.append({"name": name, "asset": item.asset, "kind": "builtin",
                                   "pos": item.pos, "rot_z_deg": item.rot_z_deg,
                                   "z": round(floor_z, 4),
                                   "size_m": [size[0], size[1], 0.012]})
                continue
            if kind == "table_lamp":
                base_z = float(item.z) if item.z is not None else floor_z
                parts = details.make_table_lamp(item.pos, collection=collection, name=f"lamp_{name}")
                shade = materials_mod_plain_principled("mat_lamp_shade", (0.95, 0.90, 0.82), 0.7)
                _add_emission(shade, strength=1.6)
                body = materials_mod_plain_principled("mat_lamp_body", (0.35, 0.33, 0.30), 0.35)
                for part in parts:
                    part.data.materials.append(shade if "shade" in part.name else body)
                    part.location.z += base_z
                lamp_lights.append((item.pos[0], item.pos[1], base_z + 0.30))
                placements.append({"name": name, "asset": item.asset, "kind": "builtin",
                                   "pos": item.pos, "rot_z_deg": item.rot_z_deg, "z": base_z,
                                   "size_m": [0.22, 0.22, 0.34]})
                continue
            # functional furniture (kitchen run, bathroom set, desk ...) built as
            # box assemblies by blender_agent/fixtures.py; the placement records
            # the created objects, which is how settle.resolve_targets finds them
            from blender_agent import fixtures as fixtures_mod
            if fixtures_mod.is_fixture(kind):
                base_z = float(item.z) if item.z is not None else floor_z
                objects = fixtures_mod.build(kind, item.pos, item.rot_z_deg, collection,
                                             name=f"fixture_{name}", base_z=base_z)
                placements.append({"name": name, "asset": item.asset, "kind": "builtin",
                                   "pos": item.pos, "rot_z_deg": item.rot_z_deg, "z": base_z,
                                   "mount": item.mount,
                                   "size_m": [round(v, 4) for v in fixtures_mod.dims(kind)],
                                   "bbox_z": [round(base_z, 4),
                                              round(base_z + fixtures_mod.dims(kind)[2], 4)],
                                   "objects": [obj.name for obj in objects]})
                continue
            raise SystemExit(f"[run_build] unknown builtin prop '{kind}'")
        asset_id = item.asset.split(":", 1)[1]
        asset_collection = assets.append_asset(libraries[asset_id], asset_id)
        asset_object = asset_collection.objects[0]
        size = [float(v) * item.scale for v in asset_object.dimensions]
        floor_z = spec.room_elevation(item.room)
        # ``z`` is a world height; without one the piece stands on its room's floor
        height = float(item.z) if item.z is not None else floor_z
        if item.mount == "ceiling":
            height = spec.room_top(item.room) - size[2]
        instance = assets.place_asset(
            asset_collection, f"furn_{name}", item.pos, item.rot_z_deg,
            height, item.scale,
        )
        if item.mount == "ceiling":
            lamp_lights.append((item.pos[0], item.pos[1], height + size[2] * 0.45))
        placements.append({
            "name": name, "asset": item.asset, "kind": "model", "instance": instance.name,
            "pos": item.pos, "rot_z_deg": item.rot_z_deg, "z": round(height, 4),
            "size_m": [round(v, 4) for v in size], "mount": item.mount,
            "bbox_z": [round(height, 4), round(height + size[2], 4)],
        })
    return {"collection": collection, "placements": placements, "lamp_lights": lamp_lights,
            "skipped": skipped}


def _add_emission(material, strength: float, color=(1.0, 0.92, 0.82)):
    """Add a soft emission to an existing Principled material (lamp shades)."""
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Emission Strength"].default_value = float(strength)
    return material


def build_lighting(spec: spec_mod.SceneSpec, scene: bpy.types.Scene, lamp_lights,
                   front3d_scene=None, report=None) -> dict:
    """Window portal area lights + a point light inside every lamp.

    A 3D-FRONT replay has no ``windows`` entries in its spec, so its portals come
    from the dataset's own ``Window`` meshes instead (``front3d.window_lights``);
    ``report`` receives the per-pane details for the manifest.
    """
    config = spec.lighting or {}
    portal_energy = float(config.get("window_portal_w", 0.0))
    portals = []
    if portal_energy > 0.0:
        if front3d_scene is not None:
            portals = front3d.window_lights(
                front3d_scene, portal_energy,
                margin=float(config.get("portal_margin", 0.02)),
                report=report,
            )
        else:
            portals = stage.build_window_portals(
                spec, portal_energy, margin=float(config.get("portal_margin", 0.02))
            )
    lamp_energy = float(config.get("lamp_w", 0.0))
    lamps = []
    if lamp_energy > 0.0 and lamp_lights:
        lamps = stage.build_lamp_lights(lamp_lights, lamp_energy,
                                        radius=float(config.get("lamp_radius", 0.08)))
    return {"portals": portals, "lamps": lamps,
            "portal_w": portal_energy, "lamp_w": lamp_energy,
            "lamp_positions": [list(map(float, p)) for p in lamp_lights]}


def resolve_spec(args, build_dir: str):
    """Load the scene spec, bridging a 3D-FRONT scene JSON when asked.

    ``--front3d-scene`` writes the bridged spec into the build directory so the
    downstream host tools (``export_scene.py``, the validator) can address the
    very same spec -- and so the replay can be repeated from the spec alone.
    """
    if args.front3d_scene:
        if not os.path.isfile(args.front3d_scene):
            raise SystemExit(f"[run_build] no such 3D-FRONT scene JSON: {args.front3d_scene}")
        scene = front3d.parse_scene(args.front3d_scene, model_root=args.front3d_model_root)
        spec_dict = front3d.to_scene_spec(
            scene,
            model_root=args.front3d_model_root,
            texture_root=args.front3d_texture_root,
            camera_source=args.front3d_camera_source,
            eye_height=args.front3d_eye_height,
            wall_thickness=args.front3d_wall_thickness,
            max_cameras=args.front3d_max_cameras,
        )
        spec_path = os.path.join(build_dir, "front3d_scene_spec.json")
        with open(spec_path, "w", encoding="utf-8") as handle:
            json.dump(spec_dict, handle, indent=2)
            handle.write("\n")
        print(f"[run_build] bridged 3D-FRONT scene -> {spec_path}")
        print(f"[run_build] {front3d.summary(scene, args.front3d_model_root, args.front3d_texture_root)}")
        spec = spec_mod.parse_spec(spec_dict, source_path=spec_path)
    else:
        if not args.spec:
            raise SystemExit("[run_build] pass --spec <scene_spec.json> or --front3d-scene <scene.json>")
        spec = spec_mod.load_spec(args.spec)
        spec_path = os.path.abspath(args.spec)

    front3d_ctx = None
    if spec.front3d:
        block = spec.front3d
        model_root = args.front3d_model_root or block.get("model_root")
        # the CLI wins over what the bridged spec recorded, and the resolved root
        # is written back so the Blender-side shell import parses with the same one
        block["model_root"] = model_root
        block["texture_root"] = args.front3d_texture_root or block.get("texture_root")
        if not model_root:
            raise SystemExit(
                "[run_build] the spec has a 'front3d' block but no model root -- pass "
                "--front3d-model-root <3D-FUTURE-model>"
            )
        front3d_ctx = {"model_root": model_root, "models": {}, "report": {},
                       "block": block, "spec_path": spec_path}
    return spec, spec_path, front3d_ctx


def main(argv) -> int:
    args = parse_args(argv)
    build_dir = os.path.abspath(args.out)
    os.makedirs(build_dir, exist_ok=True)

    if args.denoise_only:
        return run_denoise_pass(build_dir, prefilter=args.denoise_prefilter, quality=args.denoise_quality)

    spec, spec_path, front3d_ctx = resolve_spec(args, build_dir)

    if args.prepare_assets_only:
        # Building a library resets the current file and leaves a couple hundred
        # MB behind, so the render process should start from a clean slate.
        libraries = prepare_assets(spec, args.cache, force=args.rebuild_assets,
                                   resolution=args.tex_resolution)
        print(f"[run_build] asset libraries ready: {len(libraries)} models ({args.tex_resolution} textures)")
        return 0

    started = time.time()
    selftest_report = None
    if args.selftest:
        print("[run_build] running ERP orientation self-test ...")
        selftest_report = render_selftest.run_orientation_selftest(
            build_dir, width=args.width, height=args.height
        )
        status = "PASS" if selftest_report["pass"] else "FAIL"
        print(f"[run_build] orientation self-test: {status}")
        for marker in selftest_report["markers"]:
            print(
                f"[run_build]   {marker['marker']:>3}  predicted=({marker['predicted_pixel'][0]:.2f},"
                f"{marker['predicted_pixel'][1]:.2f})  detected=({marker['detected_pixel'][0]:.2f},"
                f"{marker['detected_pixel'][1]:.2f})  pixel_delta={marker['pixel_delta']:.2f}"
                f"  angle_error_px={marker['angle_error_px']:.2f}  pass={marker['pass']}"
            )
        if not selftest_report["pass"]:
            raise SystemExit("[run_build] orientation self-test FAILED - camera convention is wrong")
    else:
        # standalone run (scripts/blender_agent_run.sh runs the self-test in its
        # own process to keep this render's memory floor low): reuse its report
        report_path = os.path.join(build_dir, "selftest_report.json")
        if os.path.isfile(report_path) and os.path.getmtime(report_path) > started - 3600:
            with open(report_path, "r", encoding="utf-8") as handle:
                selftest_report = json.load(handle)
            print(
                f"[run_build] reusing orientation self-test report from {os.path.basename(report_path)}: "
                f"{'PASS' if selftest_report.get('pass') else 'FAIL'}"
            )
            if not selftest_report.get("pass"):
                raise SystemExit("[run_build] the existing self-test report is a FAIL - fix the camera convention")

    # asset libraries must be built *before* the scene: building one resets the file
    libraries = prepare_assets(spec, args.cache, force=args.rebuild_assets,
                               resolution=args.tex_resolution)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    print("[run_build] building shell geometry ...")
    if front3d_ctx is not None:
        shell_info, shell_front3d_report = front3d.build_shell(front3d_ctx["block"],
                                                               report=front3d_ctx.setdefault("shell", {}))
        front3d_ctx["shell"] = shell_front3d_report
        print(
            f"[run_build] front3d shell: {shell_front3d_report['meshes']} dataset meshes, "
            f"{shell_front3d_report['triangles']} triangles, "
            f"kinds {shell_front3d_report['by_kind']}"
        )
    else:
        shell_info = shell.build_shell(spec)
        print(
            f"[run_build] shell: {len(shell_info['floors'])} floor slabs, {len(shell_info['ceilings'])} ceilings, "
            f"{len(shell_info['walls'])} walls, {len(shell_info.get('stairs', []))} stair steps, "
            f"{len(shell_info.get('slab_openings', []))} slab openings, cutters applied: "
            f"{shell_info['cutters_applied']}"
        )
        if spec.floors:
            for storey in shell_info["storeys"]:
                print(f"[run_build]   floor '{storey['id']}': elevation {storey['elevation']:.2f} m, "
                      f"height {storey['height']:.2f} m, slab {storey['slab_thickness']:.2f} m, "
                      f"rooms {storey['rooms']}")

    flat_materials = stage.build_materials(spec)
    if front3d_ctx is not None:
        # the dataset shell brings its own materials (and its own baseboards,
        # cornices and window glass), so neither the Poly Haven scene materials
        # nor the generated trim/bevel layer run here
        textured = {}
        applied_floors = 0
        details_report = {"skirting": 0, "casings": 0, "glass": 0, "bevels": 0}
        print("[run_build] front3d shell: details layer skipped (the dataset carries its own trim)")
    else:
        textured = materials.build_scene_materials(args.cache, spec, flat_materials)
        applied_floors = materials.apply_shell_materials(shell_info, spec, textured, flat_materials)

        details_report = build_details_layer(spec, shell_info)
        if any(details_report.values()):
            print(f"[run_build] details: {details_report}")

    stage_info = stage.setup_stage(scene, spec, shell_info, materials=flat_materials,
                                   apply_materials=(front3d_ctx is None and not textured.get("source")))

    furniture_info = place_furniture(spec, libraries, textured, front3d_ctx=front3d_ctx)
    furniture_info["settle"] = settle.ground_furniture(spec, furniture_info, build_dir, shell_info)
    if furniture_info["placements"]:
        kinds = {}
        for placement in furniture_info["placements"]:
            kinds[placement["kind"]] = kinds.get(placement["kind"], 0) + 1
        print(
            f"[run_build] furniture: {len(furniture_info['placements'])} placements "
            f"({', '.join(f'{count} {kind}' for kind, count in sorted(kinds.items()))})"
        )
        for placement in furniture_info["placements"]:
            print(
                f"[run_build]   {placement['name']:<16s} {placement['asset']:<44s} "
                f"pos=({placement['pos'][0]:.2f},{placement['pos'][1]:.2f}) z={placement['z']:.3f} "
                f"rot={placement['rot_z_deg']:.1f}"
            )
    if furniture_info.get("skipped"):
        print(
            f"[run_build] furniture skipped: {len(furniture_info['skipped'])} items whose 3D-FUTURE "
            f"model folder is missing (the dataset is only partially public)"
        )
    lighting_info = build_lighting(spec, scene, furniture_info["lamp_lights"],
                                   front3d_scene=shell_info.get("front3d_scene"),
                                   report=(front3d_ctx or {}).get("report"))
    if lighting_info["portals"] or lighting_info["lamps"]:
        print(
            f"[run_build] lights: {len(lighting_info['portals'])} window portals "
            f"({lighting_info['portal_w']:.0f} W), {len(lighting_info['lamps'])} lamp lights "
            f"({lighting_info['lamp_w']:.0f} W)"
        )

    dome_radius = stage_info["dome_radius"]
    dome_center = stage_info["dome_center"]
    for camera in spec.cameras:
        distance = sum((camera.pos[i] - dome_center[i]) ** 2 for i in range(3)) ** 0.5
        max_depth = distance + dome_radius
        if max_depth >= spec_mod.DEPTH_MAX_METERS:
            raise SystemExit(
                f"[run_build] camera '{camera.name}' can see {max_depth:.1f} m, beyond the "
                f"uint16 depth range ({spec_mod.DEPTH_MAX_METERS:.1f} m at scale {spec_mod.DEPTH_SCALE:.0f})"
            )

    panorama.configure_render(
        scene,
        width=args.width,
        height=args.height,
        samples=args.samples,
        threads=args.threads,
        use_denoising=not args.no_denoise,
        view_transform=spec.style.view_transform,
        exposure=spec.style.exposure,
    )
    camera_objects = {camera.name: panorama.create_pano_camera(camera) for camera in spec.cameras}

    glb_report = None
    if args.export_glb:
        print("[run_build] exporting scene.glb for the browser 3D viewer ...")
        glb_report = export_glb.export_scene_glb(
            build_dir,
            spec,
            filepath=args.export_glb_path,
            include_environment=args.export_glb_include_environment,
        )

    camera_stats = []
    if not args.skip_render:
        for camera in spec.cameras:
            print(f"[run_build] rendering pano '{camera.name}' at {args.width}x{args.height}, {args.samples} samples ...")
            tick = time.time()
            stats = panorama.render_camera(
                scene,
                camera_objects[camera.name],
                camera,
                build_dir,
                samples=args.samples,
                save_raw_exr=args.no_denoise,
            )
            stats["seconds"] = round(time.time() - tick, 2)
            camera_stats.append(stats)
            print(
                f"[run_build]   '{camera.name}': depth {stats['depth_min']:.3f}..{stats['depth_max']:.3f} m, "
                f"invalid pixels {stats['invalid_pixels']} ({100 * (1 - stats['valid_fraction']):.4f}%), "
                f"{stats['seconds']}s"
            )
            if stats["invalid_pixels"]:
                print(f"[run_build]   WARNING: '{camera.name}' has {stats['invalid_pixels']} invalid depth pixels")

    shell_ply = os.path.join(build_dir, "shell.ply")
    shell_objects = list(shell_info["shell_objects"]) + list(stage_info["environment_objects"])
    shell_stats = panorama.export_visible_ply(shell_ply, objects=shell_objects)
    furnished_ply = os.path.join(build_dir, "furnished.ply")
    furnished_stats = panorama.export_visible_ply(furnished_ply)
    print(
        f"[run_build] exported shell.ply ({shell_stats['triangles']} tris) and "
        f"furnished.ply ({furnished_stats['triangles']} tris from {furnished_stats['instances']} instances, "
        f"{furnished_stats['bytes'] / 1e6:.1f} MB)"
    )

    manifest = {
        "kind": "blender_agent_build",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "blender_version": bpy.app.version_string,
        "seconds": round(time.time() - started, 2),
        "spec_path": spec_path,
        "spec": spec_mod.dump_manifest_spec(spec),
        "render": {
            "width": args.width,
            "height": args.height,
            "samples": args.samples,
            "threads": args.threads,
            "denoising_requested": not args.no_denoise,
            "engine": "CYCLES",
            "device": "CPU",
            "view_transform": scene.view_settings.view_transform,
            "exposure": float(scene.view_settings.exposure),
            "pixel_filter": scene.cycles.pixel_filter_type,
            "pixel_filter_width": scene.cycles.filter_width,
            "denoiser": scene.cycles.denoiser if scene.cycles.use_denoising else None,
            "depth_scale": spec_mod.DEPTH_SCALE,
        },
        "depth_scale": spec_mod.DEPTH_SCALE,
        "shell": {
            "collection": shell.SHELL_COLLECTION,
            "environment_collection": stage.ENVIRONMENT_COLLECTION,
            "ply": os.path.basename(shell_ply),
            "ply_stats": shell_stats,
            "ply_objects": [obj.name for obj in shell_objects],
            "wall_spans": len(shell_info["walls"]),
            "cutters_applied": shell_info["cutters_applied"],
            "storeys": shell_info.get("storeys", []),
            "ceiling_from_upper": shell_info.get("ceiling_from_upper", []),
            "slab_openings": shell_info.get("slab_openings", []),
            "stairs": [obj.name for obj in shell_info.get("stairs", [])],
            "ramps": [obj.name for obj in shell_info.get("ramps", [])],
            "dome_radius": dome_radius,
            "dome_center": dome_center,
        },
        "building": {
            "schema_version": spec.schema_version,
            "floors": [
                {"id": floor.id, "elevation": floor.elevation, "height": floor.height,
                 "slab_thickness": floor.slab_thickness} for floor in spec.floors
            ],
            "connectors": [
                {"kind": c.kind, "from_room": c.from_room, "to_room": c.to_room,
                 "from_floor": c.from_floor, "to_floor": c.to_floor, "width": c.width,
                 "height": c.height, "center": c.center, "run": c.run, "rise": c.rise,
                 "steps": c.steps, "landing_m": c.landing_m}
                for c in spec.connectors
            ],
            "slab_openings": [{"outline": opening.outline, "floors": opening.floors}
                              for opening in spec.slab_openings],
        },
        "visible_geometry": {
            "ply": os.path.basename(furnished_ply),
            "stats": furnished_stats,
            "collections": [shell.SHELL_COLLECTION, stage.ENVIRONMENT_COLLECTION,
                            DETAILS_COLLECTION, FURNITURE_COLLECTION],
        },
        "details": details_report,
        "materials": {
            "source": textured.get("source", {}),
            "applied_floors": applied_floors,
            "textured": textured.get("wall") is not None,
        },
        "lighting": {
            "window_portals": [obj.name for obj in lighting_info["portals"]],
            "portal_watts": lighting_info["portal_w"],
            "lamp_lights": [obj.name for obj in lighting_info["lamps"]],
            "lamp_watts": lighting_info["lamp_w"],
            "lamp_positions": lighting_info["lamp_positions"],
            "sun_energy": float(spec.style.sun_energy),
            "sky_strength": float(spec.style.sky_strength),
        },
        "furniture": furniture_info["placements"],
        "asset_cache": os.path.relpath(args.cache, REPO_ROOT),
        "asset_tex_resolution": args.tex_resolution,
        "cameras": camera_stats,
        "room_adjacency": spec.room_adjacency(),
        "camera_adjacency": spec.camera_adjacency(),
        "orientation_selftest": selftest_report,
        "orientation_selftest_inline": bool(args.selftest),
        "scene3d_glb": glb_report,
    }
    if front3d_ctx is not None:
        shell_report = front3d_ctx.get("shell", {})
        manifest["front3d"] = {
            "scene_uid": front3d_ctx["block"].get("scene_uid"),
            "scene_json": front3d_ctx["block"].get("scene_json"),
            "model_root": front3d_ctx["model_root"],
            "texture_root": front3d_ctx["block"].get("texture_root"),
            "wall_height_m": shell_report.get("wall_height_m"),
            "room_functions": shell_report.get("room_functions"),
            "category_functions": shell_report.get("category_functions"),
            "parse_warnings": front3d_ctx["block"].get("parse_warnings", []),
            "skipped_furniture": front3d_ctx["block"].get("skipped_furniture", []),
            "reassigned_furniture": front3d_ctx["block"].get("reassigned_furniture", []),
            "models": front3d_ctx.get("report", {}),
            "furniture_not_on_disk": furniture_info.get("skipped", []),
        }
    manifest_path = os.path.join(build_dir, "build_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[run_build] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    sys.exit(main(argv))
