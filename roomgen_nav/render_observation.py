"""Blender-side perspective RGB-D renderer for the PointNav observation backend.

This file runs **inside headless Blender** (Cycles CPU) and is driven by
``roomgen_nav/observation_blender.py``.  It is not importable outside Blender
(``bpy`` / ``mathutils``).

Protocol (all files are JSON, written atomically with ``os.replace``):

    <workdir>/ready.json      scene loaded, coordinates and Blender version
    <workdir>/request.json    one frame request (pose, resolution, output paths)
    <workdir>/response.json   {"ok": true, "metadata": {...}} or an error
    <workdir>/stop            clean shutdown

Coordinates
-----------
Requests carry the pose in **glTF / Habitat Y-up metres**:

* ``position = (x, y, z)`` with +Y up, the frame of ``scene.glb`` and of
  ``scene3d_manifest.json``;
* ``yaw`` degrees about +Y, ``pitch`` degrees with positive looking up.

Two yaw conventions are supported:

* ``manifest`` (default): yaw 0 looks along ``+X`` and +90 along ``-Z``,
  matching ``scene3d_manifest.json``, ``roomgen_nav.env`` and
  ``blender_agent/export_glb.blender_heading_to_gltf``;
* ``habitat``: yaw 0 looks along ``-Z`` and +90 looks along ``-X``
  (right-handed rotation about +Y, matching Habitat-Sim's agent frame).

The Blender frame is Z-up metres with ``p_blender = (x, -z, y)`` (the inverse
of ``export_glb.blender_to_gltf``'s ``p_gltf = (x, z, -y)``).  The camera is a
plain pinhole (no distortion):

* image axes: +X right, +Y up, camera looks along ``-Z`` (OpenGL / Habitat);
* intrinsics ``fx = fy = 0.5 * width / tan(hfov / 2)``, principal point at
  ``(width / 2, height / 2)`` in continuous pixel coordinates, i.e. pixel
  ``(u, v)`` samples ``(u + 0.5, v + 0.5)``;
* depth is measured by ray casting pixel centres against the evaluated scene,
  so it is exact geometry rather than a filtered render pass.  Two maps are
  written: ``depth`` = z-depth along the camera forward axis and ``distance`` =
  Euclidean ray length, both float32 metres with ``0`` marking "no hit".
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Matrix, Vector

DEFAULT_HFOV_DEG = 90.0
DEFAULT_SAMPLES = 8
MAX_DEPTH_M = 1000.0


# --------------------------------------------------------------------------- #
# coordinate helpers (kept in sync with observation_blender.py)
# --------------------------------------------------------------------------- #
def gltf_to_blender(point):
    """glTF / Habitat (x, y, z) metres -> Blender (x, -z, y)."""
    return Vector((float(point[0]), -float(point[2]), float(point[1])))


def blender_to_gltf(point):
    """Blender (x, y, z) metres -> glTF / Habitat (x, z, -y)."""
    return Vector((float(point[0]), float(point[2]), -float(point[1])))


def camera_basis(yaw_deg: float, pitch_deg: float, convention: str):
    """Return (forward, right, up) unit vectors in glTF / Habitat coordinates.

    ``habitat``: yaw 0 -> -Z, +yaw rotates towards -X.
    ``manifest``: yaw 0 -> +X, +yaw rotates towards -Z.
    ``pitch`` positive looks up.  ``right`` comes from yaw alone, so the basis
    stays defined at +-90 degrees pitch.
    """
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    if convention == "manifest":
        forward = Vector((math.cos(yaw) * math.cos(pitch),
                          math.sin(pitch),
                          -math.sin(yaw) * math.cos(pitch)))
        right = Vector((math.sin(yaw), 0.0, math.cos(yaw)))
    elif convention == "habitat":
        forward = Vector((-math.sin(yaw) * math.cos(pitch),
                          math.sin(pitch),
                          -math.cos(yaw) * math.cos(pitch)))
        right = Vector((math.cos(yaw), 0.0, -math.sin(yaw)))
    else:
        raise ValueError(f"unknown yaw convention {convention!r}")
    up = right.cross(forward)
    if up.length < 1e-9:  # cannot happen with the yaw-only right vector
        raise ValueError("degenerate camera basis")
    return forward.normalized(), right.normalized(), up.normalized()


def camera_to_world(position, yaw_deg, pitch_deg, convention) -> np.ndarray:
    """4x4 OpenGL-style camera-to-world matrix in glTF / Habitat coordinates."""
    forward, right, up = camera_basis(yaw_deg, pitch_deg, convention)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = (right.x, right.y, right.z)
    matrix[:3, 1] = (up.x, up.y, up.z)
    matrix[:3, 2] = (-forward.x, -forward.y, -forward.z)
    matrix[:3, 3] = (float(position[0]), float(position[1]), float(position[2]))
    return matrix


def pinhole_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """3x3 pinhole K; pixel centres at (u + 0.5, v + 0.5)."""
    fx = 0.5 * float(width) / math.tan(math.radians(float(hfov_deg)) / 2.0)
    fy = fx  # square pixels: Blender's sensor fit derives the vertical FOV
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# scene setup
# --------------------------------------------------------------------------- #
def _set_attr(obj, name: str, value) -> None:
    try:
        setattr(obj, name, value)
    except (AttributeError, TypeError):
        pass


def load_scene(path: str) -> str:
    """Load a .glb/.gltf/.blend scene into an empty Blender file."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path)
        return "blend"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if suffix not in (".glb", ".gltf"):
        raise ValueError(f"unsupported scene format {suffix!r} ({path})")
    try:
        bpy.ops.import_scene.gltf(filepath=path)
    except AttributeError:
        bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
        bpy.ops.import_scene.gltf(filepath=path)
    return "gltf"


def configure_render(scene, samples: int) -> None:
    scene.render.engine = "CYCLES"
    _set_attr(scene.cycles, "device", "CPU")
    _set_attr(scene.cycles, "samples", max(1, min(64, int(samples))))
    _set_attr(scene.cycles, "use_denoising", False)
    _set_attr(scene.cycles, "use_adaptive_sampling", False)
    _set_attr(scene.cycles, "seed", 0)
    _set_attr(scene.cycles, "use_animated_seed", False)
    _set_attr(scene.cycles, "max_bounces", 4)
    _set_attr(scene.cycles, "diffuse_bounces", 3)
    _set_attr(scene.cycles, "glossy_bounces", 2)
    _set_attr(scene.cycles, "transmission_bounces", 2)
    _set_attr(scene.cycles, "volume_bounces", 0)
    _set_attr(scene.cycles, "transparent_max_bounces", 4)
    _set_attr(scene.render, "use_compositing", False)
    _set_attr(scene.render, "use_sequencer", False)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_file_extension = False
    scene.render.use_overwrite = True
    scene.render.use_placeholder = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.view_settings.view_transform = "Standard"
    _set_attr(scene.view_settings, "look", "None")
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"


def configure_lighting(scene):
    """Neutral ambient world plus a point light at the camera ("headlamp").

    Imported GLBs contain no lights on purpose (the web viewer brings its own),
    and an enclosed room blocks both a fixed sun and the world background, so
    the observation renderer adds a local light that moves with the camera.
    The power is chosen for 1-5 m room distances under the Standard view
    transform: near walls clip a little, far walls stay readable.
    """
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("roomgen_obs_world")
        scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.55, 0.62, 0.72, 1.0)
        background.inputs["Strength"].default_value = 0.6

    light_data = bpy.data.lights.new("roomgen_obs_headlamp", type="POINT")
    light_data.energy = 40.0
    light_data.shadow_soft_size = 0.05
    light = bpy.data.objects.new("roomgen_obs_headlamp", light_data)
    scene.collection.objects.link(light)
    return light


def add_camera(scene, hfov_deg: float):
    camera_data = bpy.data.cameras.new("roomgen_obs_camera")
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.lens = 36.0 / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    camera_data.clip_start = 0.01
    camera_data.clip_end = MAX_DEPTH_M
    camera = bpy.data.objects.new("roomgen_obs_camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def scene_bounds_gltf(scene):
    """World-space AABB of every mesh object, converted to glTF coordinates."""
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            point = blender_to_gltf(obj.matrix_world @ Vector(corner))
            for axis in range(3):
                lo[axis] = min(lo[axis], point[axis])
                hi[axis] = max(hi[axis], point[axis])
    if lo[0] == float("inf"):
        return None
    return {"min": lo, "max": hi}


# --------------------------------------------------------------------------- #
# one frame
# --------------------------------------------------------------------------- #
def write_json_atomic(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def depth_maps(scene, camera, width: int, height: int, hfov_deg: float):
    """Ray-cast pixel centres; return (z-depth, euclidean distance) float32."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera.matrix_world.translation.copy()
    rotation = camera.matrix_world.to_3x3()
    right = (rotation @ Vector((1.0, 0.0, 0.0))).normalized()
    up = (rotation @ Vector((0.0, 1.0, 0.0))).normalized()
    forward = (rotation @ Vector((0.0, 0.0, -1.0))).normalized()

    fx = 0.5 * float(width) / math.tan(math.radians(float(hfov_deg)) / 2.0)
    fy = fx
    depth = np.zeros((height, width), dtype=np.float32)
    distance = np.zeros((height, width), dtype=np.float32)
    ray_cast = scene.ray_cast
    for row in range(height):
        ndc_y = -((row + 0.5) - 0.5 * height) / fy
        lateral_y = up * ndc_y
        for column in range(width):
            ndc_x = ((column + 0.5) - 0.5 * width) / fx
            direction = forward + right * ndc_x + lateral_y
            hit, location, _normal, _index, _object, _matrix = ray_cast(
                depsgraph, origin, direction, distance=MAX_DEPTH_M)
            if hit:
                offset = location - origin
                depth[row, column] = offset.dot(forward)
                distance[row, column] = offset.length
    return depth, distance


def render_frame(scene, camera, headlight, request: dict, defaults: dict) -> dict:
    started = time.time()
    try:
        raw_resolution = request.get("resolution") or defaults["resolution"]
        width, height = int(raw_resolution[0]), int(raw_resolution[1])
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"invalid resolution {request.get('resolution')!r}")
    if width < 1 or height < 1:
        raise ValueError(f"invalid resolution {width}x{height}")
    hfov_deg = float(request.get("hfov_deg", defaults["hfov_deg"]))
    samples = int(request.get("samples", defaults["samples"]))
    convention = str(request.get("yaw_convention", defaults["yaw_convention"]))
    position = request["position"]
    yaw = float(request["yaw"])
    pitch = float(request["pitch"])
    if len(position) != 3:
        raise ValueError(f"position must be [x, y, z], got {position!r}")
    if abs(pitch) >= 90.0:
        raise ValueError("pitch must be within (-90, 90) degrees")

    configure_render(scene, samples)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    forward, right, up = camera_basis(yaw, pitch, convention)
    forward_b = gltf_to_blender(forward).normalized()
    right_b = gltf_to_blender(right).normalized()
    up_b = gltf_to_blender(up).normalized()
    rotation_b = Matrix((
        (right_b.x, up_b.x, -forward_b.x),
        (right_b.y, up_b.y, -forward_b.y),
        (right_b.z, up_b.z, -forward_b.z),
    ))
    camera.matrix_world = Matrix.Translation(gltf_to_blender(position)) @ rotation_b.to_4x4()
    headlight.location = camera.matrix_world.translation
    bpy.context.view_layer.update()

    rgb_path = os.path.abspath(request["rgb"])
    depth_path = os.path.abspath(request["depth"])
    distance_path = os.path.abspath(request["distance"])
    meta_path = os.path.abspath(request["meta"])
    for path in (rgb_path, depth_path, distance_path, meta_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    scene.render.filepath = rgb_path
    render_started = time.time()
    bpy.ops.render.render(write_still=True)
    render_seconds = time.time() - render_started
    if not os.path.isfile(rgb_path):
        raise RuntimeError(f"Cycles did not write {rgb_path}")

    depth_started = time.time()
    depth, distance = depth_maps(scene, camera, width, height, hfov_deg)
    depth_seconds = time.time() - depth_started
    np.save(depth_path, depth)
    np.save(distance_path, distance)

    # Read the camera back out of Blender instead of trusting the input math:
    # the host compares this against its own expected matrix every frame.
    matrix = camera.matrix_world
    rotation = matrix.to_3x3()
    axes = {}
    for name, local in (("right", (1, 0, 0)), ("up", (0, 1, 0)), ("forward", (0, 0, -1))):
        vector = blender_to_gltf(rotation @ Vector(local)).normalized()
        axes[name] = [float(vector.x), float(vector.y), float(vector.z)]
    extrinsics = np.eye(4, dtype=np.float64)
    for axis, name in enumerate(("right", "up", "backward")):
        vector = blender_to_gltf(rotation @ Vector(((1, 0, 0), (0, 1, 0), (0, 0, 1))[axis]))
        extrinsics[:3, axis] = (vector.x, vector.y, vector.z)
    extrinsics[:3, 3] = [float(position[0]), float(position[1]), float(position[2])]
    intrinsics = pinhole_intrinsics(width, height, hfov_deg)

    metadata = {
        "kind": "roomgen_nav_blender_observation",
        "position_gltf": [float(value) for value in position],
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "yaw_convention": convention,
        "pitch_positive": "up",
        "resolution": [width, height],
        "hfov_deg": hfov_deg,
        "samples": samples,
        "intrinsics": intrinsics.tolist(),
        "extrinsics": extrinsics.tolist(),
        "forward_gltf": axes["forward"],
        "right_gltf": axes["right"],
        "up_gltf": axes["up"],
        "camera_axes": "OpenGL/Habitat: +X right, +Y up, -Z forward; "
                       "p_world = extrinsics @ p_camera",
        "depth_convention": "z-depth along the camera forward axis, float32 metres, "
                            "0 = no hit",
        "distance_convention": "Euclidean distance from the camera centre to the hit, "
                               "float32 metres, 0 = no hit",
        "pixel_centres": "pixel (u, v) samples continuous coordinates (u + 0.5, v + 0.5)",
        "scene_asset": defaults["scene"],
        "blender_version": bpy.app.version_string,
        "render_seconds": render_seconds,
        "depth_seconds": depth_seconds,
        "total_seconds": time.time() - started,
    }
    write_json_atomic(meta_path, metadata)
    return metadata


# --------------------------------------------------------------------------- #
# worker loop
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import argparse

    argv = list(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    parser = argparse.ArgumentParser(description="Blender-side observation renderer")
    parser.add_argument("--worker", required=True, help="request/response work directory")
    parser.add_argument("--scene", required=True, help="scene.glb / scene .blend")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--hfov-deg", type=float, default=DEFAULT_HFOV_DEG)
    parser.add_argument("--yaw-convention", default="manifest",
                        choices=("habitat", "manifest"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--parent-pid", type=int, default=None,
                        help="exit when this process is orphaned")
    args = parser.parse_args(argv)

    workdir = os.path.abspath(args.worker)
    os.makedirs(workdir, exist_ok=True)
    ready_path = os.path.join(workdir, "ready.json")
    request_path = os.path.join(workdir, "request.json")
    response_path = os.path.join(workdir, "response.json")
    stop_path = os.path.join(workdir, "stop")

    try:
        scene_format = load_scene(os.path.abspath(args.scene))
        scene = bpy.context.scene
        configure_render(scene, args.samples)
        headlight = configure_lighting(scene)
        camera = add_camera(scene, args.hfov_deg)
        bounds = scene_bounds_gltf(scene)
        defaults = {
            "scene": os.path.abspath(args.scene),
            "scene_format": scene_format,
            "samples": args.samples,
            "hfov_deg": args.hfov_deg,
            "yaw_convention": args.yaw_convention,
            "resolution": [int(args.resolution[0]), int(args.resolution[1])],
        }
        write_json_atomic(ready_path, {
            "ok": True,
            "pid": os.getpid(),
            "scene": defaults["scene"],
            "scene_format": scene_format,
            "blender_version": bpy.app.version_string,
            "bounds_gltf": bounds,
            "objects": len(scene.objects),
            "samples": args.samples,
            "hfov_deg": args.hfov_deg,
            "yaw_convention": args.yaw_convention,
        })
    except Exception:
        write_json_atomic(ready_path, {
            "ok": False,
            "error": traceback.format_exc(),
        })
        traceback.print_exc()
        return 1

    while True:
        if os.path.exists(stop_path):
            break
        if args.parent_pid is not None and os.getppid() != args.parent_pid:
            break  # the host died without writing stop: do not leak Blender
        if os.path.exists(request_path):
            try:
                with open(request_path, "r", encoding="utf-8") as handle:
                    request = json.load(handle)
                os.remove(request_path)
            except (OSError, json.JSONDecodeError):
                time.sleep(0.01)
                continue
            if os.path.exists(response_path):
                try:
                    os.remove(response_path)
                except OSError:
                    pass
            try:
                metadata = render_frame(scene, camera, headlight, request, defaults)
                write_json_atomic(response_path, {"ok": True, "metadata": metadata,
                                                  "request_id": request.get("id")})
            except Exception:
                write_json_atomic(response_path, {
                    "ok": False,
                    "error": traceback.format_exc(),
                    "request_id": request.get("id") if isinstance(request, dict) else None,
                })
        time.sleep(0.01)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
