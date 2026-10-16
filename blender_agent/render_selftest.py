"""Orientation self-test: prove the rendered ERP matches the PanoWorld formula.

Renders a low-sample equirect panorama from the origin with four emissive
marker spheres placed along world +X, -X, +Y and +Z, then checks that each
marker's centroid lands on the pixel predicted by

    lon = (x + 0.5)/W * 2*pi - pi
    lat = pi/2 - (y + 0.5)/H * pi
    ray_local = (cos(lat) sin(lon), -sin(lat), cos(lat) cos(lon))
    ray_world = c2w[:3, :3] @ ray_local

(``panoworld_lrm/utils.py::compute_plucmap_pano``).  A mirrored / flipped /
rolled camera orientation shows up immediately as a multiple-pixel error.

Two markers sit on image seams and are handled explicitly:

* ``+Z`` is the pole: its cone covers the top rows across *every* column, so
  its pixel centroid is meaningless in u.  It is verified through the mean ray
  direction of the cluster instead (which is exactly +Z by symmetry).
* ``-X`` is the longitude seam: the cluster is split across both image borders
  and is therefore clustered with wrap-around neighbours and a circular u mean.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import bmesh
import bpy
import numpy as np
from mathutils import Vector

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import panorama, spec as spec_mod  # noqa: E402  (after sys.path fix)

MARKER_DIRECTIONS: Dict[str, Tuple[float, float, float]] = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
}
MARKER_DISTANCE = 2.0
MARKER_RADIUS_PX = 3.5
TOLERANCE_PX = 3.0


def predict_pixel(direction: Sequence[float], yaw_rad: float, width: int, height: int) -> Tuple[float, float]:
    """Inverse of the ERP convention: world direction -> pixel centre (u, v)."""
    rotation = np.array(spec_mod.yaw_rotation(yaw_rad), dtype=np.float64)
    local = rotation.T @ np.array(direction, dtype=np.float64)
    norm = np.linalg.norm(local)
    local = local / norm
    lon = math.atan2(local[0], local[2])
    lat = math.asin(max(-1.0, min(1.0, -local[1])))
    u = (lon + math.pi) / (2.0 * math.pi) * width - 0.5
    v = (math.pi / 2.0 - lat) / math.pi * height - 0.5
    return u, v


def pixel_ray(u: float, v: float, yaw_rad: float, width: int, height: int) -> np.ndarray:
    """Forward ERP convention: pixel coordinate -> world ray direction."""
    lon = (u + 0.5) / width * 2.0 * math.pi - math.pi
    lat = math.pi / 2.0 - (v + 0.5) / height * math.pi
    local = np.array(
        [math.cos(lat) * math.sin(lon), -math.sin(lat), math.cos(lat) * math.cos(lon)], dtype=np.float64
    )
    return np.array(spec_mod.yaw_rotation(yaw_rad), dtype=np.float64) @ local


def _wrap_delta_u(a: float, b: float, width: int) -> float:
    delta = abs(a - b) % width
    return min(delta, width - delta)


def _cluster_bright_pixels(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """8-connected clustering with wrap-around in u."""
    height, width = mask.shape
    rows, cols = np.nonzero(mask)
    pixels = set(zip(rows.tolist(), cols.tolist()))
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {p: p for p in pixels}

    def find(p):
        root = p
        while parent[root] != root:
            root = parent[root]
        while parent[p] != root:
            parent[p], p = root, parent[p]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (y, x) in pixels:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbour = (y + dy, (x + dx) % width)
                if neighbour in pixels:
                    union((y, x), neighbour)

    groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for p in pixels:
        groups.setdefault(find(p), []).append(p)
    return list(groups.values())


def _cluster_stats(pixels: Sequence[Tuple[int, int]], yaw_rad: float, width: int, height: int) -> dict:
    directions = np.array([pixel_ray(x, y, yaw_rad, width, height) for y, x in pixels], dtype=np.float64)
    mean_direction = directions.mean(axis=0)
    mean_direction = mean_direction / np.linalg.norm(mean_direction)
    angles = np.array([2.0 * math.pi * (x + 0.5) / width - math.pi for y, x in pixels], dtype=np.float64)
    mean_lon = math.atan2(np.sin(angles).mean(), np.cos(angles).mean())
    mean_u = (mean_lon + math.pi) / (2.0 * math.pi) * width - 0.5
    mean_v = float(np.mean([y + 0.5 for y, x in pixels]) - 0.5)
    u_values = [x for y, x in pixels]
    return {
        "n_pixels": len(pixels),
        "mean_direction": [float(v) for v in mean_direction],
        "mean_pixel": [float(mean_u % width), mean_v],
        "u_extent": int(max(u_values) - min(u_values) + 1),
    }


def _make_marker(name: str, direction: Sequence[float], radius: float, collection) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"marker_{name}")
    bm = bmesh.new()
    try:
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=radius)
    except TypeError:
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, diameter=radius)
    centre = Vector((direction[0], direction[1], direction[2])) * MARKER_DISTANCE
    for vertex in bm.verts:
        vertex.co += centre
    bm.to_mesh(mesh)
    bm.free()

    material = bpy.data.materials.new(f"marker_material_{name}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 50.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    mesh.materials.append(material)

    obj = bpy.data.objects.new(f"marker_{name}", mesh)
    collection.objects.link(obj)
    return obj


def run_orientation_selftest(
    out_dir: str,
    width: int = spec_mod.PANO_WIDTH,
    height: int = spec_mod.PANO_HEIGHT,
    samples: int = 8,
    yaw_rad: float = 0.0,
    tolerance_px: float = TOLERANCE_PX,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    world = bpy.data.worlds.new("selftest_world")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    scene.world = world

    panorama.configure_render(scene, width=width, height=height, samples=samples, use_denoising=False)

    camera = spec_mod.CameraSpec(name="selftest", room="", pos=[0.0, 0.0, 0.0])
    camera_obj = panorama.create_pano_camera(camera, yaw_rad=yaw_rad)
    scene.camera = camera_obj

    marker_radius = MARKER_DISTANCE * math.tan(MARKER_RADIUS_PX * 2.0 * math.pi / width)
    markers = {}
    for name, direction in MARKER_DIRECTIONS.items():
        markers[name] = _make_marker(name, direction, marker_radius, scene.collection)

    pano_path = os.path.join(out_dir, "selftest_pano.png")
    scene.render.filepath = pano_path
    bpy.ops.render.render(write_still=True)
    for obj in markers.values():
        bpy.data.objects.remove(obj, do_unlink=True)

    pixels = panorama.read_image_pixels(pano_path)
    luma = pixels[..., :3].max(axis=-1)
    clusters = _cluster_bright_pixels(luma > 0.5)
    stats = [_cluster_stats(cluster, yaw_rad, width, height) for cluster in clusters]
    stats.sort(key=lambda item: item["n_pixels"], reverse=True)

    pixel_scale = width / (2.0 * math.pi)
    results = []
    for name, direction in MARKER_DIRECTIONS.items():
        predicted_u, predicted_v = predict_pixel(direction, yaw_rad, width, height)
        best = None
        for cluster in stats:
            mean_direction = np.array(cluster["mean_direction"], dtype=np.float64)
            angle_error = math.acos(max(-1.0, min(1.0, float(np.dot(mean_direction, direction)))))
            if best is None or angle_error < best[0]:
                best = (angle_error, cluster)
        angle_error, cluster = best
        delta_pixel = math.hypot(
            _wrap_delta_u(cluster["mean_pixel"][0], predicted_u, width),
            cluster["mean_pixel"][1] - predicted_v,
        )
        angle_error_px = angle_error * pixel_scale
        degenerate = cluster["u_extent"] > 0.5 * width
        passed = bool(min(delta_pixel, angle_error_px) <= tolerance_px) and bool(angle_error_px <= tolerance_px)
        results.append(
            {
                "marker": name,
                "direction": list(direction),
                "predicted_pixel": [predicted_u, predicted_v],
                "detected_pixel": cluster["mean_pixel"],
                "detected_pixels": cluster["n_pixels"],
                "u_extent_px": cluster["u_extent"],
                "pole_or_seam_degenerate_u": bool(degenerate),
                "pixel_delta": delta_pixel,
                "angle_error_deg": math.degrees(angle_error),
                "angle_error_px": angle_error_px,
                "pass": passed,
            }
        )

    report = {
        "kind": "orientation_selftest",
        "width": width,
        "height": height,
        "samples": samples,
        "yaw_deg": math.degrees(yaw_rad),
        "tolerance_px": tolerance_px,
        "marker_radius_px": MARKER_RADIUS_PX,
        "clusters_found": len(stats),
        "pano": os.path.basename(pano_path),
        "markers": results,
        "pass": bool(all(item["pass"] for item in results)) and len(stats) == len(MARKER_DIRECTIONS),
    }
    with open(os.path.join(out_dir, "selftest_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main(argv=None) -> int:
    """Standalone entry point (``blender --python render_selftest.py -- --out DIR``).

    Running the self-test in its own Blender process keeps the memory floor of
    the following production render low, which matters on memory-capped hosts.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="render_selftest.py")
    parser.add_argument("--out", required=True, help="directory for selftest_pano.png / selftest_report.json")
    parser.add_argument("--width", type=int, default=spec_mod.PANO_WIDTH)
    parser.add_argument("--height", type=int, default=spec_mod.PANO_HEIGHT)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--tolerance-px", type=float, default=TOLERANCE_PX)
    args = parser.parse_args(argv)

    report = run_orientation_selftest(
        args.out, width=args.width, height=args.height, samples=args.samples, tolerance_px=args.tolerance_px
    )
    print(f"[render_selftest] {'PASS' if report['pass'] else 'FAIL'} ({report['clusters_found']} markers found)")
    for marker in report["markers"]:
        print(
            f"[render_selftest]   {marker['marker']:>3} predicted=({marker['predicted_pixel'][0]:.2f},"
            f"{marker['predicted_pixel'][1]:.2f}) detected=({marker['detected_pixel'][0]:.2f},"
            f"{marker['detected_pixel'][1]:.2f}) pixel_delta={marker['pixel_delta']:.2f}"
            f" angle_error_px={marker['angle_error_px']:.2f} pass={marker['pass']}"
        )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    sys.exit(main(argv))
