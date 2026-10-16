#!/usr/bin/env python3
"""Generate furniture glbs for the ``gen:`` asset scheme.

Poly Haven and MIDI-3D cover most furniture, but a category can simply run out
(few sofas/beds/chairs of a usable size, and a room dressed from a handful of
models looks repetitive).  This tool fills those gaps with *generated*
furniture and files it under ``third_party/assets/generated``, where
``blender_agent/assets.py`` picks it up through the same index/library
machinery as ``ph:``/``midi3d:`` -- a scene spec then only says::

    "furniture": [{"asset": "gen:sofa_01", "room": "living", "pos": [3.7, 3.5]}]

Every generated glb follows the asset conventions of ``assets.py`` (metres,
origin at the bottom-centre of the bounding box, the front facing -Y after the
Blender glTF import, i.e. +Z in the glTF file), so it drops into a spec exactly
like a Poly Haven model.

Backends
--------
``--backend proc``
    procedural geometry built with trimesh on the host: boxes per category
    (sofa, chair, bed, table, cabinet, stool, lamp).  No GPU, no download, no
    network -- the fallback that always works.
``--backend trellis``
    real text-to-3D with TRELLIS (``microsoft/TRELLIS-text-xlarge``).  The host
    has no GPU, so the generator runs *inside a GPU container* (``docker exec``)
    that mounts this repo and a TRELLIS checkout at the same absolute paths,
    and writes the final glbs straight into the generated cache.
``--backend auto`` (default)
    trellis when a usable container is found, proc otherwise; the chosen
    backend is printed and recorded per asset in the cache index.

Usage (host python with trimesh; ``PANOWORLD_HOSTPY`` in the scripts)::

    # procedural assets for the three gap categories
    python tools/gen_assets_trellis.py generate --backend proc

    # real TRELLIS on a GPU container (weights first, see fetch-weights)
    python tools/gen_assets_trellis.py fetch-weights
    python tools/gen_assets_trellis.py generate --backend trellis \\
        --container roomgen-trellis --classes sofa bed chair --variants 2

    # what is in the cache, and a demo spec that places it
    python tools/gen_assets_trellis.py list
    python tools/gen_assets_trellis.py emit-spec --out outputs/gen_assets/scene_spec.json

The TRELLIS paths are this host's real ones and can be overridden with
``PANOWORLD_TRELLIS_REPO`` / ``PANOWORLD_TRELLIS_PYTHON`` /
``PANOWORLD_TRELLIS_CONTAINER`` / ``PANOWORLD_TRELLIS_HF`` or the matching
flags; see the README section "Generated assets (gen:)".
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import assets  # noqa: E402  (stdlib only: safe without bpy)

GENERATED_CACHE = assets.DEFAULT_GENERATED_CACHE
DEFAULT_SHELL_SPEC = os.path.join(REPO_ROOT, "configs", "blender_agent_p1_furnished.json")
POLYHAVEN_CACHE = assets.DEFAULT_CACHE

TRELLIS_REPO = os.environ.get("PANOWORLD_TRELLIS_REPO", "/home/dataset-assist-0/TRELLIS")
TRELLIS_PYTHON = os.environ.get("PANOWORLD_TRELLIS_PYTHON",
                                "/home/dataset-assist-0/conda_envs/trellis/bin/python")
TRELLIS_CONTAINER = os.environ.get("PANOWORLD_TRELLIS_CONTAINER", "")
TRELLIS_HF_HOME = os.environ.get("PANOWORLD_TRELLIS_HF",
                                 os.path.join(REPO_ROOT, "third_party", "assets", "trellis_hf"))
TRELLIS_HF_ENDPOINT = os.environ.get("PANOWORLD_TRELLIS_HF_ENDPOINT", "https://hf-mirror.com")
TRELLIS_PIPELINE = "microsoft/TRELLIS-text-xlarge"
TRELLIS_DECODER_REPO = "JeffreyXiang/TRELLIS-image-large"
TRELLIS_DECODER_PATTERNS = (
    "ckpts/ss_dec_conv3d_16l8_fp16*",
    "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16*",
    "ckpts/slat_dec_rf_swin8_B_64l8r16_fp16*",
    "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16*",
)
TRELLIS_TEXT_ENCODER = "openai/clip-vit-large-patch14"
# the CLIP repo ships the same weights in four formats (safetensors, .bin,
# flax, tf) -- fetch only what transformers loads
TRELLIS_TEXT_ENCODER_PATTERNS = ("*.safetensors", "*.json", "*.txt", "*.model")


# --------------------------------------------------------------------------- #
# catalogue: one entry per furniture category
# --------------------------------------------------------------------------- #
# ``target_m`` is the size a generated mesh is scaled to (width x depth x height
# in metres, measured after the front/-Y convention is applied); ``prompt`` is
# the TRELLIS text prompt; ``align`` runs a horizontal PCA so the long axis of
# the mesh ends up along X (text-to-3D has no canonical yaw at all, the
# per-category ``yaw_deg`` is the manual correction on top).
CATEGORIES = {
    "sofa": {
        "prompt": "a modern three-seat fabric sofa with a low back and short wooden legs",
        "target_m": (2.10, 0.90, 0.80),
        "categories": ("furniture", "seating", "sofa"),
        "align": True,
    },
    "armchair": {
        "prompt": "a modern upholstered armchair with a curved back and wooden legs",
        "target_m": (0.90, 0.90, 0.95),
        "categories": ("furniture", "seating", "armchair"),
        "align": True,
    },
    "chair": {
        "prompt": "a modern dining chair with a wooden frame and a padded seat",
        "target_m": (0.48, 0.52, 0.95),
        "categories": ("furniture", "seating", "chair"),
        "align": True,
    },
    "stool": {
        "prompt": "a modern wooden stool with a round seat and four legs",
        "target_m": (0.40, 0.40, 0.45),
        "categories": ("furniture", "seating", "stool"),
        "align": False,
    },
    "bed": {
        "prompt": "a modern double bed with a wooden frame, a thick mattress and a fabric headboard",
        "target_m": (1.60, 2.05, 1.00),
        "categories": ("furniture", "bed"),
        "align": True,
    },
    "table": {
        "prompt": "a modern rectangular coffee table with a wooden top and slim legs",
        "target_m": (1.10, 0.60, 0.45),
        "categories": ("furniture", "table"),
        "align": True,
    },
    "cabinet": {
        "prompt": "a modern wooden sideboard cabinet with two doors and a flat top",
        "target_m": (1.20, 0.45, 0.85),
        "categories": ("furniture", "shelves", "cabinet"),
        "align": True,
    },
    "lamp": {
        "prompt": "a modern table lamp with a round fabric shade and a metal base",
        "target_m": (0.35, 0.35, 0.55),
        "categories": ("lighting", "lamp"),
        "align": False,
    },
}
DEFAULT_CLASSES = ("sofa", "bed", "chair")

# Physical plausibility of the *measured* mesh, mirroring the curation rules of
# tools/index_midi3d.py (metres, width x depth x height).
PLAUSIBILITY = {
    "sofa": lambda w, d, h: max(w, d) >= 1.3 and 0.45 <= h <= 1.2 and min(w, d) >= 0.45,
    "armchair": lambda w, d, h: max(w, d) <= 1.3 and 0.5 <= h <= 1.3 and min(w, d) >= 0.4,
    "chair": lambda w, d, h: max(w, d) <= 1.05 and 0.5 <= h <= 1.3 and min(w, d) >= 0.25,
    "bed": lambda w, d, h: 1.5 <= max(w, d) <= 2.4 and min(w, d) >= 0.8 and h <= 1.6,
    "table": lambda w, d, h: 0.5 <= max(w, d) <= 2.4 and 0.25 <= h <= 1.15,
    "cabinet": lambda w, d, h: 0.55 <= h <= 2.5 and max(w, d) >= 0.3,
    "stool": lambda w, d, h: max(w, d) <= 1.0 and h <= 1.2,
    "lamp": lambda w, d, h: max(w, d, h) <= 2.2,
}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def rel(path: str, root: str = REPO_ROOT) -> str:
    """``root``-relative path for messages, absolute when it lives outside the repo."""
    path = os.path.abspath(path)
    if path == root or path.startswith(root + os.sep):
        return path[len(root) + 1:]
    return path


def asset_id_for(category: str, variant: int) -> str:
    return f"{category}_{variant:02d}"


def glb_relpath(asset_id: str) -> str:
    return os.path.join("models", asset_id, f"{asset_id}.glb")


# --------------------------------------------------------------------------- #
# mesh helpers (trimesh; shared by both backends)
# --------------------------------------------------------------------------- #
def _trimesh():
    try:
        import trimesh
    except ImportError:  # pragma: no cover - documented requirement
        raise SystemExit("[gen_assets] trimesh is required: it builds the proc assets and "
                         "finalises (scale/orient/measure) every generated glb")
    return trimesh


def _rotation(trimesh, angle_rad: float, axis) -> list:
    return trimesh.transformations.rotation_matrix(angle_rad, axis)


def load_parts(path: str) -> list:
    """Load a glb/gltf as a list of meshes; the scene graph is baked into them."""
    trimesh = _trimesh()
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        parts = loaded.dump(concatenate=False)
    else:
        parts = [loaded]
    return [part for part in parts if isinstance(part, trimesh.Trimesh) and len(part.faces)]


def bounds_of(parts) -> tuple:
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for part in parts:
        for corner in part.bounds:
            for i in range(3):
                lo[i] = min(lo[i], float(corner[i]))
                hi[i] = max(hi[i], float(corner[i]))
    return lo, hi


def apply_transform(parts, matrix) -> None:
    for part in parts:
        part.apply_transform(matrix)


def to_design_space(parts) -> None:
    """glTF (Y up, front +Z) -> the repo's design frame (Z up, front -Y)."""
    trimesh = _trimesh()
    apply_transform(parts, _rotation(trimesh, math.radians(90.0), (1.0, 0.0, 0.0)))


def to_gltf_space(parts) -> None:
    trimesh = _trimesh()
    apply_transform(parts, _rotation(trimesh, math.radians(-90.0), (1.0, 0.0, 0.0)))


def principal_yaw_deg(parts) -> float:
    """Angle of the dominant horizontal axis (the long axis of a sofa/bed/chair)."""
    import numpy as np

    points = np.vstack([part.vertices[:, :2] for part in parts])
    points = points - points.mean(axis=0)
    if len(points) < 8:
        return 0.0
    values, vectors = np.linalg.eigh(np.cov(points.T))
    principal = vectors[:, int(np.argmax(values))]
    return math.degrees(math.atan2(float(principal[1]), float(principal[0])))


def finalise_parts(parts, category: str, target_m=None, align=None, yaw_deg=None) -> dict:
    """Scale/orient/centre meshes in design space -> (report dict).

    Scaling is uniform on the *largest* target dimension, so the generated
    shape keeps its own proportions and only its size is fixed.  The asset then
    sits on ``z = 0`` with its bounding box centred on X/Y, exactly like a Poly
    Haven model after ``assets.normalise_asset``.
    """
    entry = CATEGORIES.get(category, {})
    if target_m is None:
        target_m = tuple(entry.get("target_m", (1.0, 1.0, 1.0)))
    if align is None:
        align = bool(entry.get("align", False))
    if yaw_deg is None:
        yaw_deg = float(entry.get("yaw_deg", 0.0))

    trimesh = _trimesh()
    lo, hi = bounds_of(parts)
    measured = [hi[i] - lo[i] for i in range(3)]
    report = {"measured_before_m": [round(v, 4) for v in measured], "target_m": list(target_m)}

    if align:
        angle = principal_yaw_deg(parts)
        if abs(angle) > 1e-3:
            apply_transform(parts, _rotation(trimesh, math.radians(-angle), (0.0, 0.0, 1.0)))
        report["pca_yaw_deg"] = round(-angle, 2)
        lo, hi = bounds_of(parts)  # the aligned extents, not the rotated AABB
        measured = [hi[i] - lo[i] for i in range(3)]
        # a bed is longer than it is wide: put the principal axis on the target's
        # longer ground axis (X for a sofa, Y for a bed) instead of always on X
        if target_m[1] > target_m[0] * 1.15 and measured[0] >= measured[1]:
            apply_transform(parts, _rotation(trimesh, math.radians(90.0), (0.0, 0.0, 1.0)))
            lo, hi = bounds_of(parts)
            measured = [hi[i] - lo[i] for i in range(3)]
            report["depth_axis_swap"] = True
        report["measured_aligned_m"] = [round(v, 4) for v in measured]

    longest = max(measured)
    if longest <= 1e-6:
        raise ValueError(f"{category}: generated mesh has no extent")
    factor = max(target_m) / longest
    apply_transform(parts, [[factor, 0, 0, 0], [0, factor, 0, 0], [0, 0, factor, 0], [0, 0, 0, 1]])
    report["uniform_scale"] = round(factor, 5)

    if yaw_deg:
        apply_transform(parts, _rotation(trimesh, math.radians(yaw_deg), (0.0, 0.0, 1.0)))
    report["yaw_deg"] = round(yaw_deg, 2)

    lo, hi = bounds_of(parts)
    offset = [-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]]
    apply_transform(parts, trimesh.transformations.translation_matrix(offset))
    lo, hi = bounds_of(parts)
    report["size_m"] = [round(hi[i] - lo[i], 4) for i in range(3)]
    return report


def export_glb(parts, path: str) -> dict:
    """Write the parts as one glb (glTF axes: Y up, front +Z)."""
    trimesh = _trimesh()
    scene = trimesh.Scene()
    for index, part in enumerate(parts):
        scene.add_geometry(part, node_name=f"part_{index}", geom_name=f"part_{index}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blob = scene.export(file_type="glb")
    with open(path, "wb") as handle:
        handle.write(blob)
    return {"glb": path, "bytes": len(blob)}


def measure_glb(path: str) -> dict:
    """Bounding box of a glb as Poly Haven style ``[width, depth, height]`` mm."""
    trimesh = _trimesh()
    parts = load_parts(path)
    if not parts:
        raise ValueError(f"{path}: no geometry")
    lo, hi = bounds_of(parts)
    size = [hi[i] - lo[i] for i in range(3)]
    return {
        "api_dimensions_mm": [round(size[0] * 1000.0, 2), round(size[2] * 1000.0, 2),
                              round(size[1] * 1000.0, 2)],
        "size_m": [round(size[0], 4), round(size[2], 4), round(size[1], 4)],
        "vertices": int(sum(len(part.vertices) for part in parts)),
        "faces": int(sum(len(part.faces) for part in parts)),
    }


def plausibility_report(category: str, size_m) -> dict:
    rule = PLAUSIBILITY.get(category)
    ok = True if rule is None else bool(rule(*[float(v) for v in size_m]))
    entry = CATEGORIES.get(category, {})
    target = list(entry.get("target_m", []))
    aspect_error = None
    if target:
        ratios = sorted(float(v) / float(t) for v, t in zip(size_m, target))
        aspect_error = round(ratios[-1] / max(ratios[0], 1e-6), 3)
    return {"plausible": ok, "aspect_ratio_error": aspect_error}


# --------------------------------------------------------------------------- #
# proc backend: boxes per category (trimesh, host side)
# --------------------------------------------------------------------------- #
def _material(trimesh, rgb, roughness=0.85, metallic=0.0):
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=[int(round(255 * c)) for c in rgb] + [255],
        metallicFactor=float(metallic), roughnessFactor=float(roughness),
        name="proc_material",
    )


def _box(trimesh, material, size, center, yaw_deg=0.0):
    mesh = trimesh.creation.box(extents=[float(v) for v in size])
    transform = trimesh.transformations.translation_matrix([float(v) for v in center])
    if yaw_deg:
        transform = transform @ _rotation(trimesh, math.radians(yaw_deg), (0.0, 0.0, 1.0))
    mesh.apply_transform(transform)
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def build_proc_parts(category: str, target_m) -> list:
    """Procedural furniture: a small assembly of boxes, built in the design frame.

    Deliberately simple -- this backend is the no-GPU fallback, not a modelling
    package.  Shapes are built at their target size (so ``finalise_parts`` only
    re-centres them) and converted to the glTF axes the cache stores.
    """
    trimesh = _trimesh()
    width, depth, height = (float(v) for v in target_m)
    fabric = _material(trimesh, (0.58, 0.60, 0.63))
    fabric_dark = _material(trimesh, (0.46, 0.48, 0.52))
    wood = _material(trimesh, (0.36, 0.24, 0.15), roughness=0.6)
    metal = _material(trimesh, (0.55, 0.55, 0.57), roughness=0.35, metallic=0.7)
    parts = []

    if category in ("sofa", "armchair", "stool"):
        arm = max(0.10, width * 0.10) if category != "stool" else 0.0
        leg_h = 0.12 if category != "stool" else 0.0
        seat_h = height * (0.34 if category == "sofa" else 0.42)
        back_h = height - leg_h - seat_h
        inner_w = max(0.10, width - 2 * arm)
        # seat + back
        parts.append(_box(trimesh, fabric, (inner_w, depth, seat_h), (0, 0, leg_h + seat_h / 2)))
        parts.append(_box(trimesh, fabric_dark, (inner_w, depth * 0.22, back_h),
                          (0, depth / 2 - depth * 0.11, leg_h + seat_h + back_h / 2)))
        for sign in (-1.0, 1.0):
            if arm:
                parts.append(_box(trimesh, fabric, (arm, depth, height - leg_h),
                                  (sign * (width - arm) / 2, 0, leg_h + (height - leg_h) / 2)))
        if category == "sofa":
            cushion_w = inner_w / 3.0
            for index in range(3):
                x = -inner_w / 2 + cushion_w * (index + 0.5)
                parts.append(_box(trimesh, fabric_dark, (cushion_w * 0.92, depth * 0.7, seat_h * 0.3),
                                  (x, -depth * 0.05, leg_h + seat_h * 1.05)))
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                if leg_h > 0:
                    parts.append(_box(trimesh, wood, (0.06, 0.06, leg_h),
                                      (sign_x * (width / 2 - 0.06), sign_y * (depth / 2 - 0.06),
                                       leg_h / 2)))

    elif category == "chair":
        seat_h = height * 0.48
        leg_h = seat_h
        parts.append(_box(trimesh, fabric, (width, depth, 0.06), (0, 0, leg_h + 0.03)))
        parts.append(_box(trimesh, wood, (width, 0.05, height - leg_h), (0, depth / 2 - 0.025,
                                                                        leg_h + (height - leg_h) / 2)))
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                parts.append(_box(trimesh, wood, (0.04, 0.04, leg_h),
                                  (sign_x * (width / 2 - 0.03), sign_y * (depth / 2 - 0.03), leg_h / 2)))

    elif category == "table":
        top_h = 0.04
        leg_h = height - top_h
        parts.append(_box(trimesh, wood, (width, depth, top_h), (0, 0, leg_h + top_h / 2)))
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                parts.append(_box(trimesh, wood, (0.05, 0.05, leg_h),
                                  (sign_x * (width / 2 - 0.06), sign_y * (depth / 2 - 0.06), leg_h / 2)))

    elif category == "bed":
        leg_h = 0.12
        frame_h = height * 0.45
        mattress_h = height * 0.35
        headboard_h = height
        parts.append(_box(trimesh, wood, (width, depth, frame_h),
                          (0, 0, leg_h + frame_h / 2)))
        parts.append(_box(trimesh, _material(trimesh, (0.86, 0.85, 0.82)),
                          (width * 0.94, depth * 0.94, mattress_h),
                          (0, 0, leg_h + frame_h + mattress_h / 2)))
        parts.append(_box(trimesh, fabric, (width, 0.10, headboard_h - leg_h),
                          (0, depth / 2 - 0.05, leg_h + (headboard_h - leg_h) / 2)))
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                parts.append(_box(trimesh, wood, (0.07, 0.07, leg_h),
                                  (sign_x * (width / 2 - 0.06), sign_y * (depth / 2 - 0.06), leg_h / 2)))

    elif category == "cabinet":
        leg_h = 0.08
        body_h = height - leg_h
        parts.append(_box(trimesh, wood, (width, depth, body_h), (0, 0, leg_h + body_h / 2)))
        for sign in (-1.0, 1.0):
            parts.append(_box(trimesh, _material(trimesh, (0.30, 0.20, 0.13), roughness=0.5),
                              (width * 0.44, 0.02, body_h * 0.86),
                              (sign * width * 0.24, -depth / 2 - 0.01, leg_h + body_h / 2)))
            parts.append(_box(trimesh, metal, (0.03, 0.03, 0.03),
                              (sign * 0.05, -depth / 2 - 0.03, leg_h + body_h * 0.5)))
        for sign in (-1.0, 1.0):
            parts.append(_box(trimesh, wood, (0.05, 0.05, leg_h),
                              (sign * (width / 2 - 0.06), -depth / 2 + 0.05, leg_h / 2)))

    elif category == "lamp":
        base_h = 0.03
        shade_h = height * 0.40
        stem_h = height - base_h - shade_h
        parts.append(_box(trimesh, metal, (width * 0.55, depth * 0.55, base_h), (0, 0, base_h / 2)))
        parts.append(_box(trimesh, metal, (0.025, 0.025, stem_h), (0, 0, base_h + stem_h / 2)))
        parts.append(_box(trimesh, _material(trimesh, (0.93, 0.89, 0.80), roughness=0.9),
                          (width * 0.85, depth * 0.85, shade_h),
                          (0, 0, base_h + stem_h + shade_h / 2)))

    else:
        raise SystemExit(f"[gen_assets] the proc backend has no builder for '{category}' "
                         f"(known: {', '.join(sorted(CATEGORIES))})")

    for index, part in enumerate(parts):
        part.metadata["name"] = f"{category}_{index}"
    to_gltf_space(parts)  # both backends hand the cache glTF-axis meshes
    return parts


# --------------------------------------------------------------------------- #
# cache bookkeeping
# --------------------------------------------------------------------------- #
def read_generated_index(cache_dir: str) -> dict:
    index = assets.load_index(cache_dir)
    return {asset_id: entry for asset_id, entry in index["models"].items()
            if entry.get("source") == "generated"}


def write_asset(cache_dir: str, asset_id: str, category: str, parts, backend: str,
                provenance: dict, extra: dict = None) -> dict:
    """Finalise -> export glb -> drop the stale library -> measure -> update index.json."""
    trimesh = _trimesh()
    to_design_space(parts)  # backend meshes arrive in glTF axes
    facts = finalise_parts(parts, category, target_m=provenance.get("target_m"),
                           align=provenance.get("align"), yaw_deg=provenance.get("yaw_deg"))
    to_gltf_space(parts)
    glb_rel = glb_relpath(asset_id)
    path = os.path.join(cache_dir, glb_rel)
    wrote = export_glb(parts, path)
    # a regenerated glb invalidates the library .blend built from the old one --
    # ensure_library() would otherwise keep serving the previous mesh
    for library in sorted(glob.glob(os.path.join(cache_dir, "library", f"{asset_id}.*.blend"))):
        os.remove(library)
        print(f"[gen_assets] dropped stale library {rel(library, REPO_ROOT)}", flush=True)
    measured = measure_glb(path)
    del trimesh

    size_m = measured["size_m"]
    plausibility = plausibility_report(category, size_m)
    entry = assets.record_generated_model(
        cache_dir, asset_id, glb_rel, measured["api_dimensions_mm"],
        CATEGORIES.get(category, {}).get("categories", ("furniture",)),
        name=f"Generated {category} ({backend})", extra={
            "backend": backend,
            "category": category,
            "generated_utc": utc_now(),
            "prompt": provenance.get("prompt"),
            "seed": provenance.get("seed"),
            "source": "generated",
            "generator": "tools/gen_assets_trellis.py",
            "finalise": facts,
            "plausible": plausibility["plausible"],
            "aspect_ratio_error": plausibility["aspect_ratio_error"],
            **(extra or {}),
        },
    )
    return {
        "id": asset_id, "category": category, "backend": backend, "glb": glb_rel,
        "bytes": wrote["bytes"], "size_m": size_m, "faces": measured["faces"],
        "vertices": measured["vertices"], "finalise": facts,
        "plausible": plausibility["plausible"], "aspect_ratio_error": plausibility["aspect_ratio_error"],
        "license": entry["license"],
    }


# --------------------------------------------------------------------------- #
# trellis backend
# --------------------------------------------------------------------------- #
def docker(*argv, capture=True, check=True):
    command = ["docker", *argv]
    if capture:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if check and result.returncode != 0:
            raise SystemExit(f"[gen_assets] {' '.join(command)} failed ({result.returncode}):\n{result.stdout.strip()}")
        return result.stdout.strip()
    return subprocess.run(command, check=check)


def running_containers() -> list:
    try:
        out = docker("ps", "--format", "{{.Names}}")
    except (SystemExit, FileNotFoundError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def container_has(container: str, path: str) -> bool:
    try:
        return subprocess.run(["docker", "exec", container, "test", "-e", path],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except FileNotFoundError:
        return False


def find_container(python: str, repo: str, explicit: str = None) -> str:
    """A running container that can see the TRELLIS checkout at the same path."""
    if explicit:
        return explicit
    candidates = running_containers()
    for name in candidates:
        if container_has(name, python) and container_has(name, repo):
            return name
    return ""


def hf_env(hf_home: str, endpoint: str) -> dict:
    """HF cache env for the container.

    ``HUGGINGFACE_HUB_CACHE`` must be set as well: the image ships one that
    points at its own read-only ``/models`` cache and it wins over ``HF_HOME``
    inside huggingface_hub.
    """
    return {
        "HF_HOME": hf_home,
        "HUGGINGFACE_HUB_CACHE": os.path.join(hf_home, "hub"),
        "HF_ENDPOINT": endpoint,
    }


def trellis_env(repo: str, hf_home: str, endpoint: str, cuda_visible: str = None) -> dict:
    env = {
        "PYTHONPATH": repo,
        "MPLCONFIGDIR": "/tmp/mpl-gen-assets",
        "SPCONV_ALGO": "native",
        **hf_env(hf_home, endpoint),
    }
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    return env


def run_in_container(container: str, python: str, repo: str, argv, env: dict, log_path: str) -> int:
    """``docker exec`` the tool inside the container, streaming its output."""
    command = ["docker", "exec"]
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    command += [container, python, os.path.join(repo, "tools", "gen_assets_trellis.py"), *argv]
    print(f"[gen_assets] container: {' '.join(command)}")
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(f"  [trellis] {line.rstrip()}")
            log.write(line)
        status = process.wait()
    print(f"[gen_assets] container finished with {status} in {time.time() - started:.1f}s "
          f"(log: {rel(log_path, REPO_ROOT)})")
    return status


def staging_dir(cache_dir: str) -> str:
    """Where plans/reports/logs live: ``<cache>/.trellis``."""
    path = os.path.join(cache_dir, ".trellis")
    os.makedirs(path, exist_ok=True)
    return path


def write_plan(cache_dir: str, jobs, args) -> str:
    path = os.path.join(staging_dir(cache_dir), f"plan-{time.strftime('%Y%m%d-%H%M%S')}.json")
    plan = {
        "kind": "gen_assets_trellis_plan",
        "created_utc": utc_now(),
        "repo": REPO_ROOT,
        "cache": cache_dir,
        "trellis_repo": args.trellis_repo,
        "hf_home": args.hf_home,
        "pipeline": TRELLIS_PIPELINE,
        "simplify": args.simplify,
        "texture_size": args.texture_size,
        "jobs": jobs,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")
    return path


def trellis_run(plan_path: str) -> int:
    """Inside the GPU container: TRELLIS text-to-3D -> final glbs -> report."""
    with open(plan_path, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("trellis_repo") and plan["trellis_repo"] not in sys.path:
        sys.path.insert(0, plan["trellis_repo"])
    from trellis.pipelines import TrellisTextTo3DPipeline
    from trellis.utils import postprocessing_utils

    import torch

    sys.path.insert(0, plan["trellis_repo"])
    cache_dir = plan.get("cache", GENERATED_CACHE)
    print(f"[trellis] torch {torch.__version__} cuda {torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)
    print(f"[trellis] loading {plan['pipeline']} ...", flush=True)
    started = time.time()
    pipeline = TrellisTextTo3DPipeline.from_pretrained(plan["pipeline"])
    pipeline.cuda()
    print(f"[trellis] pipeline ready in {time.time() - started:.1f}s", flush=True)

    reports = []
    for job in plan["jobs"]:
        asset_id = job["id"]
        tick = time.time()
        record = {"id": asset_id, "category": job["category"], "prompt": job["prompt"],
                  "seed": job.get("seed"), "backend": "trellis"}
        try:
            print(f"[trellis] {asset_id}: generating seed {job.get('seed')} ...", flush=True)
            outputs = pipeline.run(job["prompt"], seed=job.get("seed"))
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0], outputs["mesh"][0],
                simplify=float(plan.get("simplify", 0.95)),
                texture_size=int(plan.get("texture_size", 1024)),
                verbose=False,
            )
            write = write_asset(
                cache_dir, asset_id, job["category"], [glb], "trellis",
                provenance={"prompt": job["prompt"], "seed": job.get("seed"),
                            "target_m": job.get("target_m"), "align": job.get("align"),
                            "yaw_deg": job.get("yaw_deg")},
                extra={"trellis_pipeline": plan["pipeline"], "texture_size": plan.get("texture_size"),
                       "device": torch.cuda.get_device_name(0)},
            )
            record.update(write)
            record["status"] = "ok"
            del outputs, glb
            torch.cuda.empty_cache()
        except Exception as error:  # keep going: one bad prompt must not lose the run
            record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            print(f"[trellis] {asset_id} FAILED: {error}", flush=True)
        record["seconds"] = round(time.time() - tick, 2)
        reports.append(record)
        print(f"[trellis] {asset_id}: {record['status']} in {record['seconds']}s", flush=True)

    report = {
        "kind": "gen_assets_trellis_report",
        "backend": "trellis",
        "created_utc": utc_now(),
        "pipeline": plan["pipeline"],
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seconds": round(time.time() - started, 2),
        "assets": reports,
    }
    out = os.path.join(cache_dir, ".trellis", f"report-{os.path.basename(plan_path)}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"[trellis] report: {out}", flush=True)
    return 0 if all(item["status"] == "ok" for item in reports) else 1


def fetch_weights(args) -> int:
    """Download the TRELLIS checkpoints into ``--hf-home``.

    In-container is the recommended path on this host: the host cgroup is 8 GiB
    and a multi-GB download gets OOM-killed there (exit 137), while the GPU
    container has its own budget and is where the weights are consumed.
    """
    repos = [(TRELLIS_PIPELINE, None), (TRELLIS_DECODER_REPO, list(TRELLIS_DECODER_PATTERNS)),
             (TRELLIS_TEXT_ENCODER, list(TRELLIS_TEXT_ENCODER_PATTERNS))]
    print(f"[gen_assets] HF cache: {args.hf_home} (endpoint {args.hf_endpoint})")
    if args.container:
        staging = staging_dir(GENERATED_CACHE)
        env = {"MPLCONFIGDIR": "/tmp/mpl-gen-assets", **hf_env(args.hf_home, args.hf_endpoint)}
        return run_in_container(args.container, args.trellis_python, REPO_ROOT,
                                ["fetch-weights", "--hf-home", args.hf_home,
                                 "--hf-endpoint", args.hf_endpoint], env,
                                os.path.join(staging, "fetch_weights.log"))
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("[gen_assets] huggingface_hub is missing on this python; run fetch-weights "
                         "with --container <gpu container> instead")
    os.environ["HF_HOME"] = args.hf_home
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.makedirs(args.hf_home, exist_ok=True)
    for repo, patterns in repos:
        path = snapshot_download(repo, allow_patterns=patterns)
        print(f"[gen_assets] ready: {repo} -> {path}")
    print("[gen_assets] all TRELLIS weights cached")
    return 0


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_generate(args) -> int:
    cache_dir = os.path.abspath(args.cache)
    classes = [name.strip() for name in args.classes if name.strip()]
    unknown = [name for name in classes if name not in CATEGORIES]
    if unknown:
        raise SystemExit(f"[gen_assets] unknown class(es): {', '.join(unknown)} "
                         f"(known: {', '.join(sorted(CATEGORIES))})")

    backend = args.backend
    container = ""
    if backend in ("auto", "trellis"):
        container = find_container(args.trellis_python, args.trellis_repo, args.container or TRELLIS_CONTAINER)
        if backend == "auto":
            backend = "trellis" if container else "proc"
            print(f"[gen_assets] backend auto -> {backend}"
                  + (f" (container {container})" if container else
                     " (no GPU container with the TRELLIS checkout found)"))
    if backend == "trellis" and not container:
        raise SystemExit("[gen_assets] --backend trellis needs a GPU container that mounts the repo and "
                         f"the TRELLIS checkout ({args.trellis_repo}); pass --container <name>")

    variants = args.variants if args.variants > 0 else (2 if backend == "trellis" else 1)
    if backend == "proc" and args.variants <= 0:
        variants = 1

    index = read_generated_index(cache_dir)
    jobs, skipped = [], []
    for category in classes:
        entry = CATEGORIES[category]
        for variant in range(1, variants + 1):
            asset_id = asset_id_for(category, variant)
            existing = index.get(asset_id)
            if existing is not None and not args.force:
                skipped.append({"id": asset_id, "reason": f"cached ({existing.get('backend')})"})
                continue
            jobs.append({
                "id": asset_id,
                "category": category,
                "prompt": entry["prompt"],
                "seed": int(args.seed) + variant - 1,
                "target_m": list(entry["target_m"]),
                "align": bool(entry.get("align", False)),
                "yaw_deg": float(entry.get("yaw_deg", 0.0)) + float(args.yaw),
            })
    if skipped:
        for item in skipped:
            print(f"[gen_assets] skip {item['id']}: {item['reason']}")
    if not jobs:
        print("[gen_assets] nothing to do (use --force to regenerate)")
        return 0

    reports = []
    if backend == "trellis":
        staging = staging_dir(cache_dir)
        plan = write_plan(cache_dir, jobs, args)
        print(f"[gen_assets] plan: {rel(plan, REPO_ROOT)} ({len(jobs)} assets, "
              f"seed {jobs[0]['seed']}..{jobs[-1]['seed']})")
        log = os.path.join(staging, f"run-{os.path.basename(plan)[:-5]}.log")
        env = trellis_env(args.trellis_repo, args.hf_home, args.hf_endpoint,
                          args.cuda_visible_devices)
        status = run_in_container(container, args.trellis_python, REPO_ROOT,
                                  ["trellis-run", "--plan", plan], env, log)
        report_path = os.path.join(staging, f"report-{os.path.basename(plan)}")
        if os.path.isfile(report_path):
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            reports = report["assets"]
            print(f"[gen_assets] trellis: {sum(1 for r in reports if r['status'] == 'ok')}/{len(reports)} "
                  f"assets ok on {report.get('device')} in {report['seconds']}s")
        else:
            print(f"[gen_assets] no report from the container (exit {status}); see {log}")
            return status or 1
    else:
        for job in jobs:
            tick = time.time()
            print(f"[gen_assets] {job['id']}: procedural {job['category']} ...", flush=True)
            parts = build_proc_parts(job["category"], job["target_m"])
            write = write_asset(cache_dir, job["id"], job["category"], parts, "proc",
                                provenance={"prompt": None, "seed": None,
                                            "target_m": job["target_m"], "align": False,
                                            "yaw_deg": job["yaw_deg"]},
                                extra={"generator_note": "procedural boxes (no GPU)"})
            write.update({"status": "ok", "seconds": round(time.time() - tick, 2)})
            reports.append(write)

    # verify what landed on disk, then print a table
    failed = [item for item in reports if item.get("status") != "ok"]
    print()
    print(f"[gen_assets] cache: {rel(cache_dir, REPO_ROOT)}")
    for item in reports:
        if item.get("status") != "ok":
            print(f"  FAILED {item['id']:12s} {item.get('error')}")
            continue
        size = ", ".join(f"{v:.2f}" for v in item["size_m"])
        print(f"  {item['id']:12s} {item['backend']:7s} {item['category']:8s} "
              f"{size} m  {item['faces']:>6d} faces  {item['bytes'] / 1e6:.2f} MB"
              + ("" if item.get("plausible", True) else "  WARNING implausible size")
              + ("" if (item.get("aspect_ratio_error") or 1.0) < 1.6 else "  WARNING odd proportions"))
    summary_path = os.path.join(cache_dir, "generation_report.json")
    summary = {"kind": "gen_assets_generation", "created_utc": utc_now(), "backend": backend,
               "assets": reports}
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        summary["history"] = previous.get("history", [])[-9:] + [previous.get("created_utc")]
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"[gen_assets] report: {rel(summary_path, REPO_ROOT)}")
    return 1 if failed and not reports else (0 if not failed else 1)


def cmd_list(args) -> int:
    cache_dir = os.path.abspath(args.cache)
    index = read_generated_index(cache_dir)
    if not index:
        print(f"[gen_assets] no generated assets in {rel(cache_dir, REPO_ROOT)}")
        return 0
    print(f"{'id':14s} {'backend':8s} {'category':10s} {'size (w,d,h) m':24s} {'glb':40s}")
    for asset_id in sorted(index):
        entry = index[asset_id]
        dims = entry.get("api_dimensions_mm") or [0, 0, 0]
        size = ", ".join(f"{v / 1000:.2f}" for v in dims)
        path = os.path.join(cache_dir, entry["gltf"])
        print(f"{asset_id:14s} {str(entry.get('backend')):8s} {str(entry.get('category')):10s} "
              f"{size:24s} {entry['gltf']:40s} {'OK' if os.path.isfile(path) else 'MISSING'}")
        if entry.get("prompt") and args.verbose:
            print(f"               prompt: {entry['prompt']!r} seed {entry.get('seed')}")
    print(f"# {len(index)} generated assets in {rel(cache_dir, REPO_ROOT)}")
    return 0


# --------------------------------------------------------------------------- #
# emit-spec: a demo scene that places the generated furniture
# --------------------------------------------------------------------------- #
PH_EXTRAS = (
    {"asset": "ph:modern_ceiling_lamp_01", "room": "living", "pos": [3.70, 2.85],
     "mount": "ceiling", "name": "ceiling_lamp"},
    {"asset": "ph:modern_ceiling_lamp_01", "room": "bedroom", "pos": [7.50, 1.55],
     "mount": "ceiling", "name": "ceiling_lamp_bed"},
    {"asset": "ph:potted_plant_01", "room": "living", "pos": [0.55, 3.55], "rot_z_deg": 0,
     "name": "plant_tall"},
    {"asset": "ph:potted_plant_02", "room": "bedroom", "pos": [8.55, 3.15], "rot_z_deg": 0,
     "name": "plant_bedroom"},
)
PH_NIGHTSTAND = {"asset": "ph:painted_wooden_nightstand", "name": "nightstand"}
WALL_INSET = 0.06  # inner face of a wall: wall_thickness / 2


def polyhaven_ids(cache_dir: str) -> set:
    index = assets.load_index(cache_dir)
    return {asset_id for asset_id, entry in index["models"].items()
            if "1k" in (entry.get("gltf_variants") or {})}


def best_generated(cache_dir: str, category: str) -> dict:
    """The most plausible generated asset of a category (best aspect first)."""
    candidates = [entry for entry in read_generated_index(cache_dir).values()
                  if entry.get("category") == category and entry.get("plausible", True)]
    if not candidates:
        return {}
    candidates.sort(key=lambda entry: (entry.get("aspect_ratio_error") or 9.9, entry.get("id")))
    return candidates[0]


def dims_of(entry: dict) -> tuple:
    dims = entry.get("api_dimensions_mm") or [1000.0, 1000.0, 1000.0]
    return tuple(float(value) / 1000.0 for value in dims)


def build_demo_furniture(cache_dir: str, polyhaven: set) -> tuple:
    """Place the generated assets in the shipped two-room shell; returns (items, notes)."""
    items, notes = [], []
    sofa = best_generated(cache_dir, "sofa")
    table = best_generated(cache_dir, "table")
    chair = best_generated(cache_dir, "chair")
    armchair = best_generated(cache_dir, "armchair")
    cabinet = best_generated(cache_dir, "cabinet")
    bed = best_generated(cache_dir, "bed")

    if sofa:
        width, depth, _ = dims_of(sofa)
        items.append({"asset": f"gen:{sofa['id']}", "room": "living",
                      "pos": [3.70, 4.10 - WALL_INSET - depth / 2.0], "rot_z_deg": 0, "name": "sofa"})
        items.append({"asset": "builtin:rug", "room": "living", "pos": [3.40, 2.45],
                      "rot_z_deg": 0, "size": [2.6, 1.5], "name": "rug"})
    if table:
        items.append({"asset": f"gen:{table['id']}", "room": "living",
                      "pos": [3.60, 2.45], "rot_z_deg": 90, "name": "coffee_table"})
    if chair:
        items.append({"asset": f"gen:{chair['id']}", "room": "living",
                      "pos": [3.85, 0.95], "rot_z_deg": 180, "name": "chair_east"})
        items.append({"asset": f"gen:{chair['id']}", "room": "living",
                      "pos": [1.35, 1.20], "rot_z_deg": 180, "name": "chair_west"})
    elif armchair:
        items.append({"asset": f"gen:{armchair['id']}", "room": "living",
                      "pos": [3.60, 1.15], "rot_z_deg": 180, "name": "armchair"})
    if cabinet:
        width, depth, height = dims_of(cabinet)
        items.append({"asset": f"gen:{cabinet['id']}", "room": "living",
                      "pos": [WALL_INSET + depth / 2.0, 2.05], "rot_z_deg": 90, "name": "cabinet"})
    if bed:
        width, depth, height = dims_of(bed)
        items.append({"asset": f"gen:{bed['id']}", "room": "bedroom",
                      "pos": [8.25, 1.43], "rot_z_deg": 180, "name": "bed"})
        if PH_NIGHTSTAND["asset"].split(":", 1)[1] in polyhaven:
            items.append({**PH_NIGHTSTAND, "room": "bedroom", "pos": [7.15, 0.75], "rot_z_deg": 180})
    for extra in PH_EXTRAS:
        if extra["asset"].split(":", 1)[1] in polyhaven:
            items.append(dict(extra))
        else:
            notes.append(f"{extra['asset']} is not in the Poly Haven cache, skipped")
    return items, notes


def build_check_index(cache_dir: str) -> str:
    """Merged/full asset index + the generated entries, for blender_agent.checks."""
    index = assets.load_index(POLYHAVEN_CACHE)
    merged = os.path.join(os.path.dirname(POLYHAVEN_CACHE), "merged_index.json")
    if os.path.isfile(merged):
        with open(merged, "r", encoding="utf-8") as handle:
            index = json.load(handle)
    index.setdefault("models", {})
    index["models"].update(read_generated_index(cache_dir))
    path = os.path.join(cache_dir, ".check_index.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"version": 2, "models": index["models"], "textures": index.get("textures", {})},
                  handle, indent=1, sort_keys=True)
    return path


def cmd_emit_spec(args) -> int:
    cache_dir = os.path.abspath(args.cache)
    with open(args.shell_spec, "r", encoding="utf-8") as handle:
        shell = json.load(handle)
    furniture, notes = build_demo_furniture(cache_dir, polyhaven_ids(POLYHAVEN_CACHE))
    generated = sorted({item["asset"] for item in furniture if item["asset"].startswith("gen:")})
    if not generated:
        raise SystemExit("[gen_assets] no generated assets in the cache; run "
                         "'generate --classes sofa bed chair' first")

    spec = {key: shell[key] for key in
            ("unit", "wall_height", "wall_thickness", "rooms", "openings", "windows",
             "cameras", "style", "materials", "details", "lighting") if key in shell}
    spec["_comment"] = [
        "demo spec generated by tools/gen_assets_trellis.py emit-spec:",
        f"the shell (rooms/openings/cameras/materials) is {rel(args.shell_spec, REPO_ROOT)}",
        f"furniture comes from the generated cache {rel(cache_dir, REPO_ROOT)}:",
        "  " + ", ".join(generated),
        "each gen:<id> is a glb built by the generator and normalised to the asset",
        "conventions of blender_agent/assets.py (metres, bottom-centre origin, front -Y).",
    ]
    if notes:
        spec["_comment"] += ["skipped extras: " + "; ".join(notes)]
    spec["furniture"] = furniture

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)
        handle.write("\n")
    print(f"[gen_assets] spec: {rel(out, REPO_ROOT)} "
          f"({len(furniture)} furniture items, {len(generated)} generated)")

    assets_report = {
        "kind": "gen_assets_spec_assets",
        "created_utc": utc_now(),
        "spec": out,
        "generated": {asset_id: {key: entry.get(key) for key in
                                 ("backend", "category", "prompt", "seed", "api_dimensions_mm",
                                  "gltf", "generated_utc", "plausible", "aspect_ratio_error")}
                      for asset_id, entry in sorted(read_generated_index(cache_dir).items())
                      if f"gen:{asset_id}" in generated},
        "notes": notes,
    }
    assets_path = os.path.join(os.path.dirname(out), os.path.splitext(os.path.basename(out))[0]
                               + ".assets.json")
    with open(assets_path, "w", encoding="utf-8") as handle:
        json.dump(assets_report, handle, indent=2)
        handle.write("\n")
    print(f"[gen_assets] provenance: {rel(assets_path, REPO_ROOT)}")

    if args.check:
        from blender_agent import checks

        index_path = build_check_index(cache_dir)
        report = checks.check_file(out, checks.load_assets_index(index_path))
        report_path = os.path.join(os.path.dirname(out), "checks_report.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        for entry in report["checks"]:
            print(f"[gen_assets]   [{'PASS' if entry['pass'] else 'FAIL'}] {entry['name']}")
            for violation in entry.get("violations", []):
                print(f"[gen_assets]          - {violation}")
        # informational by default: the shipped P1 spec has violations of its own
        # (the camera sits 0.3 m from the bed), so a spec that renders and
        # validates is not the same thing as a perfect layout score
        print(f"[gen_assets] placement checks: {'PASS' if report['pass'] else 'FAIL'} "
              f"({len(report.get('violations', []))} violations) -> {rel(report_path, REPO_ROOT)}")
        if args.strict and not report["pass"]:
            return 1
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_trellis_options(parser) -> None:
    parser.add_argument("--trellis-repo", default=TRELLIS_REPO,
                        help="TRELLIS checkout (env PANOWORLD_TRELLIS_REPO)")
    parser.add_argument("--trellis-python", default=TRELLIS_PYTHON,
                        help="python of the TRELLIS conda env (env PANOWORLD_TRELLIS_PYTHON)")
    parser.add_argument("--container", default="",
                        help="GPU container name (env PANOWORLD_TRELLIS_CONTAINER; default: auto-detect)")
    parser.add_argument("--hf-home", default=TRELLIS_HF_HOME, help="HF cache for the TRELLIS weights")
    parser.add_argument("--hf-endpoint", default=TRELLIS_HF_ENDPOINT, help="HF mirror for the download")
    parser.add_argument("--cuda-visible-devices", default=None,
                        help="CUDA_VISIBLE_DEVICES inside the container (default: keep the container's)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate furniture glbs for the 'gen:' asset scheme",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=GENERATED_CACHE, help="generated asset cache directory")
    sub = parser.add_subparsers(dest="command", required=True)

    p_generate = sub.add_parser("generate", help="generate assets for the requested classes")
    p_generate.add_argument("--backend", default="auto", choices=["auto", "proc", "trellis"])
    p_generate.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    p_generate.add_argument("--variants", type=int, default=0,
                            help="assets per class (default: 2 for trellis, 1 for proc)")
    p_generate.add_argument("--seed", type=int, default=1, help="TRELLIS seed of the first variant")
    p_generate.add_argument("--yaw", type=float, default=0.0, help="extra yaw (deg) on every asset")
    p_generate.add_argument("--simplify", type=float, default=0.95, help="TRELLIS mesh simplification")
    p_generate.add_argument("--texture-size", type=int, default=1024, help="TRELLIS texture size")
    p_generate.add_argument("--force", action="store_true", help="regenerate assets already cached")
    add_trellis_options(p_generate)
    p_generate.set_defaults(func=cmd_generate)

    p_run = sub.add_parser("trellis-run", help="(inside the GPU container) run a plan and export glbs")
    p_run.add_argument("--plan", required=True)
    p_run.set_defaults(func=lambda args: trellis_run(args.plan))

    p_fetch = sub.add_parser("fetch-weights", help="download the TRELLIS checkpoints")
    add_trellis_options(p_fetch)
    p_fetch.set_defaults(func=fetch_weights)

    p_list = sub.add_parser("list", help="show the generated cache")
    p_list.add_argument("--verbose", action="store_true", help="include prompts + seeds")
    p_list.set_defaults(func=cmd_list)

    p_spec = sub.add_parser("emit-spec", help="write a demo scene spec that places the generated assets")
    p_spec.add_argument("--out", required=True, help="scene spec to write")
    p_spec.add_argument("--shell-spec", default=DEFAULT_SHELL_SPEC,
                        help="spec whose shell (rooms/cameras/materials) is reused")
    p_spec.add_argument("--no-check", dest="check", action="store_false",
                        help="skip the blender_agent.checks placement check")
    p_spec.add_argument("--strict", action="store_true",
                        help="exit non-zero when the placement check reports violations")
    p_spec.set_defaults(func=cmd_emit_spec, check=True, strict=False)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
