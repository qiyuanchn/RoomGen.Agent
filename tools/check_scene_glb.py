#!/usr/bin/env python3
"""Re-read a viewer scene.glb without Blender and check it is usable.

    /opt/conda/envs/diffaero/bin/python tools/check_scene_glb.py outputs/blender_agent/p1_settled

Checks (exit code 1 on any failure):
  * the GLB loads (trimesh), has meshes, triangles and materials
  * the bounding box is room sized (metres) and the floor sits near y = 0
  * scene3d_manifest.json bounds agree with the geometry trimesh sees
    (this is what proves the Blender -> glTF coordinate conversion in
    blender_agent/export_glb.py matches the exporter)
  * every panorama camera in the manifest sits inside the geometry
    (rays from the pose hit surfaces in most directions)
  * the triangle count is in the same ballpark as furnished.ply, i.e. the
    furniture made it through the export
  * every GLB mesh node carries (or inherits from a collection-instance parent)
    the four semantic ids and they match ``scene3d_manifest.json``
    (``semantics.objects``); legacy GLBs without the table are only warned about
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from blender_agent.export_glb import read_glb_document, verify_glb_semantics  # noqa: E402

GLB_CANDIDATES = ("scene.glb", os.path.join("build", "scene.glb"))
RAY_DIRECTIONS = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (0.577, 0.577, 0.577), (-0.577, 0.577, 0.577), (0.577, -0.577, 0.577), (0.577, 0.577, -0.577),
)
MAX_RAY_METERS = 20.0


def find_glb(scene_dir: str) -> str:
    scene_dir = os.path.abspath(os.path.expanduser(scene_dir))
    if os.path.isfile(scene_dir):
        return scene_dir
    for candidate in GLB_CANDIDATES:
        path = os.path.join(scene_dir, candidate)
        if os.path.isfile(path):
            return path
    raise SystemExit(f"[check_scene_glb] no scene.glb under {scene_dir} (tried {', '.join(GLB_CANDIDATES)})")


def material_names(geometry: dict) -> set:
    names = set()
    for name, mesh in geometry.items():
        material = getattr(getattr(mesh, "visual", None), "material", None)
        if material is None:
            names.add(f"<default:{name}>")
            continue
        names.add(getattr(material, "name", None) or f"<material:{id(material)}>")
    return names


def scene_triangles(scene: trimesh.Scene, geometry: dict) -> int:
    """Triangles the GLB actually renders: every node-mesh reference counted."""
    counts = {name: int(mesh.faces.shape[0]) for name, mesh in geometry.items()}
    total = 0
    for node in scene.graph.nodes_geometry:
        try:
            _transform, names = scene.graph[node]
        except (KeyError, ValueError):
            continue
        if names is None:
            continue
        for name in ([names] if isinstance(names, str) else names):
            total += counts.get(name, 0)
    return total


def ray_triangles(triangles: np.ndarray, origin, direction, max_distance: float) -> tuple:
    """Möller-Trumbore for one ray against every triangle (no rtree/embree needed).

    trimesh's ray API wants the optional ``rtree`` module, so the few camera
    probes this checker needs are done in plain numpy.
    """
    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0
    pvec = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    valid = np.abs(det) > 1e-12
    if not valid.any():
        return 0, math.inf
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1)
    v = (qvec @ direction) * inv_det
    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
    hit = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-4) & (t < max_distance)
    if not hit.any():
        return 0, math.inf
    return int(np.count_nonzero(hit)), float(t[hit].min())


def camera_inside(triangles: np.ndarray, position) -> tuple:
    """Cast the probe rays from a camera pose; count those that hit geometry."""
    origin = np.asarray(position, dtype=float)
    hits, nearest = 0, math.inf
    for direction in RAY_DIRECTIONS:
        count, distance = ray_triangles(
            triangles, origin, np.asarray(direction, dtype=float), MAX_RAY_METERS)
        if count:
            hits += 1
            nearest = min(nearest, distance)
    return hits, nearest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a viewer scene.glb without Blender")
    parser.add_argument("scene_dir", help="scene/build directory containing scene.glb")
    parser.add_argument("--json", dest="json_out", help="write the report to this JSON file")
    parser.add_argument("--max-extent", type=float, default=40.0, help="fail above this bbox size (m)")
    parser.add_argument("--min-glb-ply-ratio", type=float, default=0.8,
                        help="minimum GLB/furnished.ply triangle ratio")
    args = parser.parse_args(argv)

    glb_path = find_glb(args.scene_dir)
    print(f"[check_scene_glb] {glb_path} ({os.path.getsize(glb_path) / 1e6:.1f} MB)")
    loaded = trimesh.load(glb_path, force="scene")
    if not isinstance(loaded, trimesh.Scene):
        loaded = trimesh.Scene(loaded)

    geometry = {name: mesh for name, mesh in loaded.geometry.items() if isinstance(mesh, trimesh.Trimesh)}
    triangles = int(sum(mesh.faces.shape[0] for mesh in geometry.values()))
    materials = material_names(geometry)
    bounds = loaded.bounds
    size = (bounds[1] - bounds[0]) if bounds is not None else np.zeros(3)

    failures, warnings = [], []
    if not geometry:
        failures.append("no meshes in the GLB")
    if triangles == 0:
        failures.append("no triangles in the GLB")
    if not materials:
        failures.append("no materials in the GLB")
    if size is None or max(size) > args.max_extent:
        failures.append(f"bounding box {size} exceeds {args.max_extent} m")
    if size[1] < 1.0:
        failures.append(f"vertical extent {size[1]:.2f} m is below one storey")
    if abs(float(bounds[0][1])) > 3.0:
        warnings.append(f"floor is at y = {bounds[0][1]:.2f} m (expected near 0)")

    print(f"[check_scene_glb] {len(geometry)} meshes, {triangles} triangles, "
          f"{len(materials)} materials, {len(loaded.graph.nodes)} nodes")
    print(f"[check_scene_glb] bounds min {np.round(bounds[0], 3).tolist()} "
          f"max {np.round(bounds[1], 3).tolist()}")
    print(f"[check_scene_glb] size {np.round(size, 2).tolist()} m "
          f"(W x H x D in glTF axes)")

    report = {
        "glb": glb_path,
        "bytes": os.path.getsize(glb_path),
        "meshes": len(geometry),
        "triangles": triangles,
        "materials": sorted(materials),
        "bounds": {"min": [float(v) for v in bounds[0]], "max": [float(v) for v in bounds[1]]},
        "size": [float(v) for v in size],
        "cameras": [],
        "failures": failures,
        "warnings": warnings,
    }

    manifest_path = os.path.join(os.path.dirname(glb_path), "scene3d_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("bounds"):
            expected = manifest["bounds"]
            delta = max(abs(expected["min"][i] - float(bounds[0][i])) for i in range(3))
            delta = max(delta, max(abs(expected["max"][i] - float(bounds[1][i])) for i in range(3)))
            report["bounds_delta_m"] = delta
            if delta > 0.02:
                failures.append(f"manifest bounds differ from the GLB by {delta:.3f} m "
                                f"(coordinate conversion bug in export_glb.py?)")
            else:
                print(f"[check_scene_glb] manifest bounds match the geometry (max delta {delta * 1000:.1f} mm)")

        stats = manifest.get("stats") or {}
        if stats.get("triangles_unique"):
            delta = abs(triangles - int(stats["triangles_unique"]))
            report["triangles_delta"] = delta
            if delta > max(64, 0.02 * triangles):
                failures.append(f"manifest says {stats['triangles_unique']} unique triangles, "
                                f"trimesh reads {triangles}")
            else:
                print(f"[check_scene_glb] manifest triangle count matches ({triangles}, delta {delta})")

        triangles_array = loaded.to_geometry().triangles
        for camera in manifest.get("cameras", []):
            position = camera.get("position")
            if not position:
                continue
            hits, nearest = camera_inside(triangles_array, position)
            inside = hits >= 4 and nearest < MAX_RAY_METERS
            entry = {"name": camera.get("name"), "position": position, "rays_hit": hits,
                     "nearest_hit_m": None if math.isinf(nearest) else round(nearest, 3),
                     "inside": inside}
            report["cameras"].append(entry)
            status = "ok" if inside else "FAIL"
            print(f"[check_scene_glb] camera {entry['name']}: {hits}/10 rays hit within "
                  f"{MAX_RAY_METERS:g} m, nearest {entry['nearest_hit_m']} m -> {status}")
            if not inside:
                failures.append(f"camera {entry['name']} at {position} looks outside the geometry")
        report["cameras_checked"] = len(report["cameras"])

        document = None
        try:
            document = read_glb_document(glb_path)
        except Exception as exc:  # noqa: BLE001 - report the parse problem, keep checking
            failures.append(f"could not parse the GLB JSON chunk for semantics: {exc}")
        if document is not None:
            sem_failures, sem_warnings, sem_report = verify_glb_semantics(document, manifest)
            failures.extend(sem_failures)
            warnings.extend(sem_warnings)
            report["semantics"] = sem_report
            if manifest.get("semantics"):
                print(f"[check_scene_glb] semantics: {sem_report.get('instances', 0)} instances, "
                      f"{sem_report.get('objects', 0)} objects, "
                      f"{sem_report.get('categories', 0)} categories, "
                      f"{sem_report.get('resolved_mesh_nodes', 0)}/"
                      f"{sem_report.get('mesh_nodes', 0)} mesh nodes resolved to extras")
            else:
                print("[check_scene_glb] semantics: absent (legacy GLB, extras not checked)")
    else:
        warnings.append(f"no {os.path.basename(manifest_path)} next to the GLB")

    furnished = os.path.join(os.path.dirname(glb_path), "furnished.ply")
    if os.path.isfile(furnished):
        ply = trimesh.load(furnished, process=False)
        ply_triangles = int(sum(mesh.faces.shape[0] for mesh in ply.geometry.values())) \
            if isinstance(ply, trimesh.Scene) else int(ply.faces.shape[0])
        # furnished.ply counts placed instances (plus the 2 208-triangle sky
        # dome), so compare against the GLB's per-node triangle count.
        rendered = scene_triangles(loaded, geometry)
        ratio = rendered / ply_triangles if ply_triangles else 0.0
        report["glb_triangles_rendered"] = rendered
        report["furnished_ply_triangles"] = ply_triangles
        report["glb_ply_triangle_ratio"] = ratio
        print(f"[check_scene_glb] furnished.ply has {ply_triangles} triangles, "
              f"GLB nodes reference {rendered}, ratio {ratio:.3f}")
        if ratio < args.min_glb_ply_ratio:
            failures.append(f"GLB holds only {ratio:.2f} of the furnished.ply triangles "
                            f"(geometry lost in the export?)")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"[check_scene_glb] wrote {args.json_out}")

    for warning in warnings:
        print(f"[check_scene_glb] WARNING: {warning}")
    for failure in failures:
        print(f"[check_scene_glb] FAIL: {failure}")
    print("[check_scene_glb] " + ("PASS" if not failures else f"FAILED ({len(failures)} problems)"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
