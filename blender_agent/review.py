"""Low-cost visual and rule review of a generated scene package.

The numeric validators (`tools/validate_blender_agent_p0.py`, `checks.py`)
answer *geometric* questions: does the depth match the mesh, is the doorway
really open, does a furniture box stay inside its room.  A scene can pass all
of them and still be a bad sample: an empty kitchen, a sofa duplicated on top
of itself, furniture that settled floating, a camera pressed against a wall
facing away from the room, or a window blown out into a white blob.  Those are
the failures a training set notices and the numeric checks do not.

This module reviews a scene package -- ``spec.json``, ``build/`` and the
exported ``scene/viewpoints/`` -- with cheap, *computed* checks (no model, no
GPU, no Blender) and writes ``review_report.json``:

| check | reads | catches |
|---|---|---|
| `rooms_furnished` | spec | a habitable room left empty (only a lamp and a plant) |
| `functional_coverage` | spec | kitchen without a counter/stove/sink, bathroom without a toilet, bedroom without a bed, ... |
| `no_duplicate_furniture` | spec | the same asset placed twice in the same spot; monotone repeats |
| `camera_clearance` | spec | a camera in a wall, on the furniture, without a stand |
| `camera_coverage` | spec | a room no camera stands in |
| `furniture_grounding` | build manifest + settle | floating / sunk / interpenetrating furniture after settling |
| `build_matches_spec` | manifest | a stale build: the manifest was made from a different spec |
| `exposure` | rendered panoramas | blown highlight blobs, a very dark viewpoint |
| `camera_framing` | spec | *advisory*: the stand faces away from the room's furniture |
| `external_review` | `REVIEW_CMD` | anything a pluggable image reviewer reports |

Every issue carries a machine-readable ``repair`` hint, and
:func:`apply_repair` turns the solvable ones into **spec tweaks** (furnish an
empty room, move a camera stand, calm an over-lit style) -- validated against
``checks.py`` before the spec is written, so the closed loop
*review -> repair -> re-check -> re-render -> re-review* never publishes a spec
the placement checks would reject.

The review is a *quality* gate, not a correctness gate: only issues with
``"severity": "fail"`` fail a scene (`review.pass`), the rest are warnings that
ride along in the report.  ``--strict`` promotes warnings to failures.  The
gates are the computed defects a downstream training set cannot use (an empty
habitable room, a kitchen without a kitchen, furniture that settled through the
floor, a build made from a different spec, a panorama that is one white blob);
the warnings are the quality headroom (a camera in the comfort band along a
wall, a viewpoint facing the empty side of its room, a window that clips a
large part of the frame).

Command line::

    python blender_agent/review.py <outdir> [--json review_report.json]
    python blender_agent/review.py <outdir> --spec-only        # no build/render reads
    python blender_agent/review.py <outdir> --apply-repair     # patch spec.json

Exit codes: ``0`` pass, ``1`` review failed, ``2`` usage/IO error; for
``--apply-repair``: ``0`` spec changed, ``1`` nothing to change, ``2`` error.

External reviewers plug in through the environment: ``REVIEW_CMD`` is split
with :func:`shlex.split` and gets the panorama image paths appended; it must
print one JSON object on stdout (``{"pass": bool, "issues": [...],
"suggestions": [...]}``) and is run with a timeout.  A reviewer that crashes or
prints junk is recorded as ``external.status: "error"`` -- it never fails the
scene on its own, so a broken hook cannot poison a batch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import checks as checks_mod  # noqa: E402
from blender_agent import fixtures as fixtures_mod  # noqa: E402
from blender_agent import layout_gen as layout_mod  # noqa: E402

DEFAULT_REPORT_NAME = "review_report.json"
SCHEMA_VERSION = 1

# --- thresholds ------------------------------------------------------------ #
# Camera quality: `checks.py` rejects a stand closer than 0.10 m to a wall;
# the review is about *quality*, so the comfort band starts earlier.
WALL_HUG_FAIL_M = 0.15
WALL_HUG_WARN_M = 0.35
# Share of the room's furniture volume in front of the stand (the camera looks
# along +X; see `spec.py`).  Advisory: an equirect camera sees the whole sphere,
# so "furniture behind the stand" costs nothing but the centre of the frame --
# the metric still says which way the viewpoint faces.
SUBJECT_WARN_SHARE = 0.35
SUBJECT_NONE_SHARE = 0.0
SUBJECT_MIN_ITEMS = 3
# Exposure: share of the frame with luminance >= CLIP_LEVEL, and the largest
# connected clipped blob (measured on a coarse mask).  Shipped scenes run
# 0-25 % clip / 0-17 % blob, so 35 % / 25 % means "the view is really gone".
CLIP_LEVEL = 0.98
CLIP_WARN_FRACTION = 0.15
CLIP_FAIL_FRACTION = 0.35
CLIP_BLOB_WARN = 0.08
CLIP_BLOB_FAIL = 0.25
DARK_WARN_LUMINANCE = 0.08
# Grounding (settled manifest): tolerances of `settle.py` are 1 mm margin and
# 20 mm lift, so a floor piece more than 2 cm up or 2 cm down is broken.
FLOAT_WARN_M, FLOAT_FAIL_M = 0.02, 0.05
SUNK_WARN_M, SUNK_FAIL_M = 0.005, 0.02
# Duplicates: same asset, same room, centres closer than this.
DUPLICATE_GAP_M = 0.35
MONOTONY_COUNT = 4

HABITABLE = ("living_room", "bedroom", "study", "kitchen", "dining")
SERVICE = ("bathroom", "entry", "storage", "laundry")
TRANSIT = ("corridor", "balcony")

#: asset categories that dress a room but do not *furnish* it
DECOR_CATEGORIES = frozenset({"lighting", "plants", "potted plants", "nature", "decorative",
                              "wall decoration", "books", "containers", "props", "rug"})
#: generated props (`details.py`) with no index entry
DECOR_ASSETS = frozenset({"builtin:rug", "builtin:table_lamp"})

#: room type -> the `builtin:` part that fixes an empty room of that type
REPAIR_FIXTURE = {
    "kitchen": "builtin:kitchen_cabinet",
    "bathroom": "builtin:bath_toilet",
    "dining": "builtin:dining_table",
    "study": "builtin:study_desk",
    "entry": "builtin:entry_shoe_cabinet",
    "storage": "builtin:storage_shelf",
    "laundry": "builtin:laundry_washer",
    "bedroom": "builtin:wardrobe",
    "living_room": "builtin:dining_table",
}
#: richer first choice: an index asset from `layout_gen.ASSET_SLOTS`
REPAIR_SLOT = {"living_room": "sofa", "bedroom": "bed", "study": "bookshelf"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Optional[str]) -> Optional[dict]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _is_decor(asset: str, index: Dict[str, dict]) -> bool:
    """True for items that dress a room (lamp, plant, art, rug, basket)."""
    if asset in DECOR_ASSETS:
        return True
    return bool(_asset_categories(asset, index)) and \
        _asset_categories(asset, index) <= DECOR_CATEGORIES


def _asset_categories(asset: str, index: Dict[str, dict]) -> set:
    scheme, _, asset_id = asset.partition(":")
    if scheme == "builtin":
        part = fixtures_mod.FIXTURE_PARTS.get(asset_id)
        if part is not None:
            return {part["category"]} if part.get("category") else set()
        return {"rug"}
    entry = index.get(asset_id) or {}
    return {str(category) for category in (entry.get("categories") or ())}


def _fixture_parts(assets: Iterable[str]) -> set:
    return {asset.split(":", 1)[1] for asset in assets
            if asset.startswith("builtin:") and asset.split(":", 1)[1] in fixtures_mod.FIXTURE_PARTS}


def _rect_of_polygon(polygon: Sequence[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    box = (min(xs), min(ys), max(xs), max(ys))
    if len(polygon) != 4:
        return None
    for x, y in polygon:                        # only the generator's rectangles
        if x not in (box[0], box[2]) or y not in (box[1], box[3]):
            return None
    return box


def _item_volume(box: checks_mod.ItemBox) -> float:
    return max(1e-4, (box.aabb[2] - box.aabb[0]) * (box.aabb[3] - box.aabb[1]) * (box.z[1] - box.z[0]))


def _slot_asset(slot: str, index: Dict[str, dict]) -> Optional[str]:
    """Deterministic first match of a `layout_gen` furniture slot, or ``None``."""
    spec_slot = layout_mod.ASSET_SLOTS.get(slot)
    if spec_slot is None:
        return None
    for name in sorted(index):
        entry = index[name]
        dims = entry.get("dims_m")
        if not dims:
            continue
        categories = set(entry.get("categories") or ())
        if all(category in categories for category in spec_slot["categories"]) and spec_slot["dims"](dims):
            return entry.get("ref") or f"ph:{name}"
    for name in spec_slot["names"]:
        entry = index.get(name)
        if entry and entry.get("dims_m") and spec_slot["dims"](entry["dims_m"]):
            return entry.get("ref") or f"ph:{name}"
    return None


def _repair_asset(room_type: str, index: Dict[str, dict]) -> Optional[str]:
    slot = REPAIR_SLOT.get(room_type)
    if slot:
        asset = _slot_asset(slot, index)
        if asset:
            return asset
    return REPAIR_FIXTURE.get(room_type)


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
class Artifacts:
    """Everything the review reads, with a single place that knows the paths."""

    def __init__(self, outdir: str, spec_path: Optional[str] = None,
                 manifest_path: Optional[str] = None) -> None:
        self.outdir = os.path.abspath(outdir)
        self.spec_path = os.path.abspath(spec_path) if spec_path else os.path.join(self.outdir, "spec.json")
        self.manifest_path = os.path.abspath(manifest_path) if manifest_path \
            else os.path.join(self.outdir, "build", "build_manifest.json")
        self.raw_spec = _load_json(self.spec_path)
        self.manifest = _load_json(self.manifest_path)
        self.settle_report = _load_json(os.path.join(self.outdir, "build", "settle_report.json"))
        self.checks_report = _load_json(os.path.join(self.outdir, "checks_report.json"))
        self.layout_report = _load_json(os.path.join(self.outdir, "layout_report.json"))
        self.viewpoints = os.path.join(self.outdir, "scene", "viewpoints")

    @property
    def has_spec(self) -> bool:
        return isinstance(self.raw_spec, dict)

    @property
    def has_build(self) -> bool:
        return isinstance(self.manifest, dict)

    def panoramas(self) -> List[Tuple[str, str]]:
        """``(camera name, pano png)`` for every exported viewpoint."""
        out: List[Tuple[str, str]] = []
        if not os.path.isdir(self.viewpoints):
            return out
        for name in sorted(os.listdir(self.viewpoints)):
            view = os.path.join(self.viewpoints, name)
            if not os.path.isdir(view):
                continue
            for entry in sorted(os.listdir(view)):
                if entry.startswith("panoImage_") and entry.endswith(".png"):
                    out.append((name, os.path.join(view, entry)))
                    break
        return out

    def depth_maps(self) -> Dict[str, str]:
        build = os.path.join(self.outdir, "build")
        out: Dict[str, str] = {}
        if not os.path.isdir(build):
            return out
        for entry in sorted(os.listdir(build)):
            if entry.startswith("depth_") and entry.endswith(".npy"):
                out[entry[len("depth_"):-len(".npy")]] = os.path.join(build, entry)
        return out


# --------------------------------------------------------------------------- #
# checks: spec scope
# --------------------------------------------------------------------------- #
def _check(name: str, scope: str, severity: str, issues: List[dict],
           **details: object) -> dict:
    """One check entry: ``pass`` means *no issue at all*, the severity is the
    worst issue found (or the check's gate severity when it is clean)."""
    fails = [issue for issue in issues if issue.get("severity") == "fail"]
    report_severity = "fail" if fails else ("warn" if issues else severity)
    return {
        "name": name,
        "scope": scope,
        "severity": report_severity,
        "gate": severity,
        "pass": not issues,
        "status": "ok" if not issues else "issues",
        "issues": issues,
        "issues_fail": len(fails),
        "issues_warn": len(issues) - len(fails),
        "details": details,
    }


def _empty(status: str, name: str, scope: str, severity: str, reason: str) -> dict:
    return {"name": name, "scope": scope, "severity": severity, "gate": severity, "pass": True,
            "status": status, "issues": [], "issues_fail": 0, "issues_warn": 0,
            "details": {"reason": reason}}


def check_rooms_furnished(ctx: dict) -> dict:
    """A habitable room needs furniture; a lamp and a plant are not furniture."""
    items_by_room = ctx["items_by_room"]
    issues: List[dict] = []
    per_room: Dict[str, dict] = {}
    for room in ctx["spec"].rooms:
        items = items_by_room.get(room.id, [])
        substantive = [item for item in items if not _is_decor(item.asset, ctx["index"])]
        per_room[room.id] = {"type": room.type, "items": len(items), "furnished": len(substantive)}
        if room.type in TRANSIT:
            continue
        if substantive:
            continue
        if room.type in HABITABLE:
            detail = (f"habitable room '{room.id}' ({room.type}) has no furniture "
                      f"({len(items)} decorative item(s) only)")
            repair = {"kind": "add_furniture", "room": room.id, "room_type": room.type}
        else:
            detail = f"room '{room.id}' ({room.type}) holds no furniture at all"
            repair = {"kind": "add_furniture", "room": room.id, "room_type": room.type}
        issues.append({"check": "rooms_furnished", "severity": "fail", "kind": "empty_room",
                       "room": room.id, "room_type": room.type, "detail": detail, "repair": repair})
    return _check("rooms_furnished", "spec", "fail", issues, rooms=per_room)


def _functional_gaps(room_type: str, items: Sequence[checks_mod.ItemBox],
                     index: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    """``(failures, warnings)``: what a room of this type is missing."""
    assets = [item.asset for item in items]
    categories = {category for asset in assets for category in _asset_categories(asset, index)}
    parts = _fixture_parts(assets)
    fails: List[str] = []
    warns: List[str] = []
    if room_type == "kitchen":
        run = parts & {"kitchen_cabinet", "kitchen_stove", "kitchen_sink"}
        if not run:
            fails.append("no kitchen counter, stove or sink")
        else:
            if not run & {"kitchen_cabinet", "kitchen_sink"}:
                warns.append("no sink or cabinet in the kitchen run")
            if "kitchen_stove" not in run:
                warns.append("no stove in the kitchen")
    elif room_type == "bathroom":
        if "bath_toilet" not in parts:
            fails.append("no toilet")
        if not parts & {"bath_basin", "bath_shower"}:
            warns.append("no washbasin or shower")
    elif room_type == "bedroom":
        if "bed" not in categories:
            fails.append("no bed")
        if not parts & {"wardrobe"} and "shelves" not in categories:
            warns.append("no wardrobe or shelving")
    elif room_type == "living_room":
        if "seating" not in categories:
            fails.append("no seating")
        if not categories & {"table"}:
            warns.append("no table (coffee or side)")
    elif room_type == "dining":
        if not parts & {"dining_table"} and "table" not in categories:
            fails.append("no dining table")
    elif room_type == "study":
        if not parts & {"study_desk"} and "desk" not in categories:
            if "shelves" in categories:
                warns.append("no desk (the study has shelving only)")
            else:
                fails.append("no desk or shelving")
    elif room_type == "entry":
        if not parts & {"entry_shoe_cabinet"} and "wardrobe" not in parts:
            warns.append("no shoe cabinet or wardrobe")
    elif room_type == "laundry":
        if "laundry_washer" not in parts:
            warns.append("no washing machine")
    return fails, warns


def check_functional_coverage(ctx: dict) -> dict:
    """Kitchen / bathroom / bedroom / living / study / dining need their gear."""
    items_by_room = ctx["items_by_room"]
    issues: List[dict] = []
    per_room: Dict[str, dict] = {}
    for room in ctx["spec"].rooms:
        items = items_by_room.get(room.id, [])
        fails, warns = _functional_gaps(room.type, items, ctx["index"])
        per_room[room.id] = {"type": room.type, "missing": fails + [f"{w} (warn)" for w in warns]}
        for gap in fails:
            issues.append({"check": "functional_coverage", "severity": "fail",
                           "kind": f"missing_{room.type}", "room": room.id, "room_type": room.type,
                           "detail": f"room '{room.id}' ({room.type}): {gap}",
                           "repair": {"kind": "add_furniture", "room": room.id, "room_type": room.type,
                                      "need": gap}})
        for gap in warns:
            issues.append({"check": "functional_coverage", "severity": "warn",
                           "kind": f"thin_{room.type}", "room": room.id, "room_type": room.type,
                           "detail": f"room '{room.id}' ({room.type}): {gap}",
                           "repair": {"kind": "note", "room": room.id}})
    return _check("functional_coverage", "spec", "fail", issues, rooms=per_room)


def check_no_duplicate_furniture(ctx: dict) -> dict:
    """Two copies of one asset on one spot, or a room filled with the same item."""
    boxes = ctx["boxes"]
    issues: List[dict] = []
    by_room: Dict[str, List[checks_mod.ItemBox]] = {}
    for box in boxes:
        by_room.setdefault(box.room, []).append(box)
    for room_id, items in sorted(by_room.items()):
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                first, second = items[i], items[j]
                if first.asset != second.asset:
                    continue
                # a duplicate shares the spot *and* the height: five book stacks
                # on five shelves share the XY but not the Z
                overlap = min(first.z[1], second.z[1]) - max(first.z[0], second.z[0])
                if overlap <= 0.05:
                    continue
                ax = 0.5 * (first.aabb[0] + first.aabb[2])
                ay = 0.5 * (first.aabb[1] + first.aabb[3])
                bx = 0.5 * (second.aabb[0] + second.aabb[2])
                by = 0.5 * (second.aabb[1] + second.aabb[3])
                gap = math.hypot(ax - bx, ay - by)
                if gap <= DUPLICATE_GAP_M:
                    issues.append({
                        "check": "no_duplicate_furniture", "severity": "fail",
                        "kind": "duplicate_placement", "room": room_id,
                        "detail": f"'{first.name}' and '{second.name}' are both {first.asset} "
                                  f"within {gap:.2f} m in room '{room_id}'",
                        "repair": {"kind": "note", "room": room_id,
                                   "why": "duplicate placements need a re-draw of the room, not a patch"},
                    })
        counts: Dict[str, int] = {}
        for item in items:
            counts[item.asset] = counts.get(item.asset, 0) + 1
        for asset, count in sorted(counts.items()):
            if count < MONOTONY_COUNT or asset.startswith("builtin:"):
                continue
            issues.append({
                "check": "no_duplicate_furniture", "severity": "warn",
                "kind": "monotone_room", "room": room_id, "asset": asset,
                "detail": f"room '{room_id}' repeats '{asset}' {count} times",
                "repair": {"kind": "note", "room": room_id},
            })
    return _check("no_duplicate_furniture", "spec", "fail", issues,
                  rooms={room: len(items) for room, items in sorted(by_room.items())})


def camera_framing(items: Sequence[checks_mod.ItemBox], pos: Sequence[float]) -> dict:
    """How much of the room's furniture stands in front of a stand (camera looks +X)."""
    total = 0.0
    forward = 0.0
    angles: List[float] = []
    for box in items:
        if box.placement in (checks_mod.FLOOR_FLAT, checks_mod.CEILING, checks_mod.MOUNTED):
            continue
        cx = 0.5 * (box.aabb[0] + box.aabb[2])
        cy = 0.5 * (box.aabb[1] + box.aabb[3])
        angle = math.degrees(math.atan2(cy - pos[1], cx - pos[0]))
        volume = _item_volume(box)
        total += volume
        if abs(angle) <= 90.0:
            forward += volume
        angles.append(round(angle, 1))
    share = (forward / total) if total > 0 else 1.0
    return {"volume_share": round(share, 3), "items": len(angles), "azimuths_deg": sorted(angles)}


def check_camera_clearance(ctx: dict) -> dict:
    """Camera stand: inside its room, off the walls, off the furniture."""
    spec = ctx["spec"]
    rooms = {room.id: room for room in spec.rooms}
    issues: List[dict] = []
    per_camera: Dict[str, dict] = {}
    for camera in spec.cameras:
        room = rooms.get(camera.room)
        if room is None:
            issues.append({"check": "camera_clearance", "severity": "fail", "kind": "camera_room_missing",
                           "camera": camera.name, "detail": f"camera '{camera.name}' names unknown room "
                                                           f"'{camera.room}'",
                           "repair": {"kind": "note"}})
            continue
        point = (camera.pos[0], camera.pos[1], camera.pos[0], camera.pos[1])
        gap = checks_mod.wall_gap(point, room.polygon)
        nearest, nearest_gap = None, float("inf")
        for box in ctx["items_by_room"].get(camera.room, []):
            if not box.blocks_camera:
                continue
            distance = checks_mod.rect_distance(point, box.aabb)
            if distance < nearest_gap:
                nearest, nearest_gap = box.name, distance
        per_camera[camera.name] = {"room": camera.room, "wall_gap_m": round(gap, 3),
                                   "nearest_furniture": nearest,
                                   "furniture_gap_m": None if nearest is None else round(nearest_gap, 3)}
        if gap < 0.0:
            issues.append({"check": "camera_clearance", "severity": "fail", "kind": "camera_outside_room",
                           "camera": camera.name, "room": camera.room,
                           "detail": f"camera '{camera.name}' is outside room '{camera.room}'",
                           "repair": {"kind": "move_camera", "camera": camera.name, "room": camera.room}})
        elif gap < WALL_HUG_FAIL_M:
            issues.append({"check": "camera_clearance", "severity": "fail", "kind": "camera_in_wall",
                           "camera": camera.name, "room": camera.room,
                           "detail": f"camera '{camera.name}' is {gap:.2f} m from a wall",
                           "repair": {"kind": "move_camera", "camera": camera.name, "room": camera.room}})
        elif gap < WALL_HUG_WARN_M:
            issues.append({"check": "camera_clearance", "severity": "warn", "kind": "camera_hugs_wall",
                           "camera": camera.name, "room": camera.room,
                           "detail": f"camera '{camera.name}' stands {gap:.2f} m off the wall "
                                     f"(comfort {WALL_HUG_WARN_M:.2f} m)",
                           "repair": {"kind": "move_camera", "camera": camera.name, "room": camera.room}})
        if nearest is not None and nearest_gap < checks_mod.CAMERA_FURNITURE_GAP_M - 1e-9:
            issues.append({"check": "camera_clearance", "severity": "fail", "kind": "camera_in_furniture",
                           "camera": camera.name, "room": camera.room,
                           "detail": f"camera '{camera.name}' is {nearest_gap:.2f} m from '{nearest}'",
                           "repair": {"kind": "move_camera", "camera": camera.name, "room": camera.room}})
    return _check("camera_clearance", "spec", "fail", issues, cameras=per_camera)


def check_camera_framing(ctx: dict) -> dict:
    """Advisory: which way the stand faces relative to the room's furniture.

    A panorama sees the whole sphere, so this never fails a scene -- it is the
    quality signal behind "the centre of the frame shows a wall".
    """
    issues: List[dict] = []
    per_camera: Dict[str, dict] = {}
    for camera in ctx["spec"].cameras:
        items = [box for box in ctx["items_by_room"].get(camera.room, [])
                 if not _is_decor(box.asset, ctx["index"])]
        framing = camera_framing(items, camera.pos)
        per_camera[camera.name] = dict(framing, room=camera.room)
        if framing["items"] < SUBJECT_MIN_ITEMS:
            continue
        if framing["volume_share"] <= SUBJECT_NONE_SHARE:
            severity, kind = "warn", "subject_behind_camera"
            detail = (f"camera '{camera.name}' has all {framing['items']} furniture items in room "
                      f"'{camera.room}' behind it")
        elif framing["volume_share"] < SUBJECT_WARN_SHARE:
            severity, kind = "warn", "subject_off_to_one_side"
            detail = (f"camera '{camera.name}' sees {framing['volume_share']:.2f} of room "
                      f"'{camera.room}'s furniture volume in front")
        else:
            continue
        issues.append({"check": "camera_framing", "severity": severity, "kind": kind,
                       "camera": camera.name, "room": camera.room, "detail": detail,
                       "repair": {"kind": "move_camera", "camera": camera.name, "room": camera.room}})
    return _check("camera_framing", "spec", "warn", issues, cameras=per_camera)


def check_camera_coverage(ctx: dict) -> dict:
    """Every room a person spends time in should have a viewpoint."""
    covered = {camera.room for camera in ctx["spec"].cameras}
    issues: List[dict] = []
    for room in ctx["spec"].rooms:
        if room.id in covered or room.type in TRANSIT:
            continue
        issues.append({"check": "camera_coverage", "severity": "warn", "kind": "room_without_camera",
                       "room": room.id, "room_type": room.type,
                       "detail": f"room '{room.id}' ({room.type}) has no camera",
                       "repair": {"kind": "note", "room": room.id}})
    return _check("camera_coverage", "spec", "warn", issues, cameras=len(ctx["spec"].cameras))


# --------------------------------------------------------------------------- #
# checks: build scope
# --------------------------------------------------------------------------- #
def check_build_matches_spec(ctx: dict) -> dict:
    """The manifest must have been built from this spec (a repair invalidates it)."""
    manifest = ctx["artifacts"].manifest
    built = manifest.get("spec")
    if built is None:
        return _check("build_matches_spec", "build", "fail",
                      [{"check": "build_matches_spec", "severity": "warn", "kind": "spec_not_recorded",
                        "detail": "the build manifest carries no spec copy", "repair": {"kind": "note"}}],
                      note="manifest without 'spec'")
    if built != ctx["artifacts"].raw_spec:
        different = sorted(key for key in set(built) | set(ctx["artifacts"].raw_spec)
                           if built.get(key) != ctx["artifacts"].raw_spec.get(key))
        return _check("build_matches_spec", "build", "fail",
                      [{"check": "build_matches_spec", "severity": "fail", "kind": "stale_build",
                        "detail": "the rendered build was made from a different spec ("
                                  + ", ".join(different[:6]) + ")",
                        "repair": {"kind": "note", "why": "re-run the blender pipeline on this spec"}}],
                      differing_keys=different)
    return _check("build_matches_spec", "build", "fail", [], note="manifest spec == spec.json")


def check_furniture_grounding(ctx: dict) -> dict:
    """Floating / sunk / interpenetrating furniture, from the settled manifest."""
    items = ctx["artifacts"].manifest.get("furniture") or []
    if not items or not any((item.get("settle") or {}) for item in items):
        return _empty("not_run", "furniture_grounding", "build", "fail",
                      "the manifest has no settle data (built before settle.py, or --skip-render)")
    issues: List[dict] = []
    per_item: Dict[str, dict] = {}
    for item in items:
        settle = item.get("settle") or {}
        mode = settle.get("mode")
        gap = settle.get("bottom_gap_m")
        per_item[item.get("name") or item.get("asset")] = {
            "asset": item.get("asset"), "mode": mode, "bottom_gap_m": gap,
            "grounded": settle.get("grounded"), "penetrating": settle.get("penetrating")}
        if mode != "rest" or settle.get("flat") or item.get("mount") in ("ceiling", "wall"):
            continue                                   # held/mounted or floor finish: by design
        name = item.get("name") or item.get("asset")
        if gap is None:
            continue
        if gap > FLOAT_FAIL_M:
            severity, kind = "fail", "furniture_floating"
        elif gap > FLOAT_WARN_M:
            severity, kind = "warn", "furniture_off_floor"
        elif gap < -SUNK_FAIL_M:
            severity, kind = "fail", "furniture_sunk"
        elif gap < -SUNK_WARN_M:
            severity, kind = "warn", "furniture_sunk_a_little"
        elif settle.get("penetrating"):
            severity, kind = "fail", "furniture_penetrating"
        else:
            continue
        issues.append({"check": "furniture_grounding", "severity": severity, "kind": kind,
                       "item": name, "asset": item.get("asset"),
                       "detail": f"'{name}' ({item.get('asset')}) bottom gap {gap * 1000:.1f} mm after settling",
                       "repair": {"kind": "note", "why": "the settle pass could not ground this piece"}})
    report_checks = (ctx["artifacts"].settle_report or {}).get("checks") or {}
    return _check("furniture_grounding", "build", "fail", issues, items=per_item,
                  settle_pass=report_checks.get("pass"), settle_violations=report_checks.get("violations"))


# --------------------------------------------------------------------------- #
# checks: render scope
# --------------------------------------------------------------------------- #
def _pano_stats(path: str) -> Optional[dict]:
    """Clipped-pixel fraction, largest clipped blob and luminance of one ERP png."""
    try:
        import numpy as np
        from PIL import Image
    except Exception:                                    # pragma: no cover - host without numpy/PIL
        return None
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            step = max(1, int(math.ceil(max(width, height) / 1024.0)))
            if step > 1:
                image = image.resize((max(1, width // step), max(1, height // step)))
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except Exception:
        return None
    luminance = 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]
    clipped = luminance >= CLIP_LEVEL
    return {
        "size": [int(pixels.shape[1]), int(pixels.shape[0])],
        "clip_fraction": float(clipped.mean()),
        "largest_blob_fraction": _largest_blob_fraction(clipped),
        "mean_luminance": float(luminance.mean()),
        "p99_luminance": float(np.percentile(luminance, 99.0)),
    }


def _largest_blob_fraction(mask, target: int = 160) -> float:
    """Share of ``mask`` covered by its largest 4-connected blob (coarse mask)."""
    try:
        import numpy as np
    except Exception:                                    # pragma: no cover
        return 0.0
    height, width = mask.shape
    factor = max(1, int(round(width / float(target))))
    rows = height // factor
    cols = width // factor
    if rows < 1 or cols < 1:
        return float(mask.mean())
    coarse = mask[:rows * factor, :cols * factor].reshape(rows, factor, cols, factor).mean(axis=(1, 3)) >= 0.5
    if not coarse.any():
        return 0.0
    seen = np.zeros_like(coarse, dtype=bool)
    best = 0
    for start in np.argwhere(coarse):
        r, c = int(start[0]), int(start[1])
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < rows and 0 <= nx < cols and coarse[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        best = max(best, size)
    return float(best) / float(coarse.size)


def _depth_stats(path: str) -> Optional[dict]:
    try:
        import numpy as np

        depth = np.load(path).astype("float32")
    except Exception:
        return None
    valid = depth[depth > 0]
    if valid.size == 0:
        return None
    height = depth.shape[0]
    band = depth[height // 4: (3 * height) // 4, :]      # |latitude| <= 45 deg
    band = band[band > 0]
    return {"median_depth_m": float(np.median(valid)),
            "near_fraction_horizon": float((band < 1.0).mean()) if band.size else 0.0}


def check_exposure(ctx: dict) -> dict:
    """Blown highlights and dark viewpoints in the rendered panoramas."""
    panoramas = ctx["artifacts"].panoramas()
    if not panoramas:
        return _empty("not_run", "exposure", "render", "fail", "no exported panoramas")
    depths = ctx["artifacts"].depth_maps() if ctx.get("depth", True) else {}
    issues: List[dict] = []
    per_camera: Dict[str, dict] = {}
    for camera, path in panoramas:
        stats = _pano_stats(path)
        if stats is None:
            continue
        record = dict(stats)
        if camera in depths:
            depth = _depth_stats(depths[camera])
            if depth:
                record.update(depth)
        per_camera[camera] = {key: (round(value, 4) if isinstance(value, float) else value)
                              for key, value in record.items()}
        clip, blob, mean = stats["clip_fraction"], stats["largest_blob_fraction"], stats["mean_luminance"]
        reason = None
        if clip >= CLIP_FAIL_FRACTION or blob >= CLIP_BLOB_FAIL:
            severity, kind = "fail", "blown_out_view"
            reason = (f"{clip * 100:.1f} % of the frame is clipped, largest blob {blob * 100:.1f} %")
        elif clip >= CLIP_WARN_FRACTION or blob >= CLIP_BLOB_WARN:
            severity, kind = "warn", "bright_blob"
            reason = (f"{clip * 100:.1f} % of the frame is clipped, largest blob {blob * 100:.1f} %")
        if reason:
            issues.append({"check": "exposure", "severity": severity, "kind": kind, "camera": camera,
                           "detail": f"camera '{camera}': {reason}",
                           "repair": {"kind": "reduce_exposure", "camera": camera}})
        if mean < DARK_WARN_LUMINANCE:
            issues.append({"check": "exposure", "severity": "warn", "kind": "dark_view",
                           "camera": camera,
                           "detail": f"camera '{camera}': mean luminance {mean:.3f} "
                                     f"(viewpoint is very dark)",
                           "repair": {"kind": "add_light", "camera": camera}})
    return _check("exposure", "render", "fail", issues, cameras=per_camera)


def run_external_review(ctx: dict, command: Optional[str], timeout: float = 180.0) -> dict:
    """Run the pluggable ``REVIEW_CMD`` on the panoramas and fold in its JSON."""
    panoramas = [path for _, path in ctx["artifacts"].panoramas()]
    if not command:
        return {"status": "not_configured", "command": None, "issues": []}
    if not panoramas:
        return {"status": "skipped_no_images", "command": command, "issues": []}
    argv = shlex.split(command)
    if not argv:
        return {"status": "error", "command": command, "issues": [],
                "error": "empty REVIEW_CMD"}
    started = time.time()
    try:
        result = subprocess.run(argv + panoramas, capture_output=True, text=True, timeout=timeout)
    except Exception as error:
        return {"status": "error", "command": command, "issues": [],
                "error": f"{type(error).__name__}: {error}"}
    payload = None
    for chunk in (result.stdout, result.stdout[result.stdout.find("{"):] if "{" in result.stdout else ""):
        try:
            payload = json.loads(chunk)
            break
        except Exception:
            continue
    if not isinstance(payload, dict):
        return {"status": "error", "command": command, "issues": [],
                "exit_code": result.returncode, "seconds": round(time.time() - started, 2),
                "error": "no JSON object on stdout",
                "stdout_tail": result.stdout[-400:], "stderr_tail": result.stderr[-400:]}
    issues = []
    for entry in payload.get("issues") or []:
        if isinstance(entry, dict):
            issue = dict(entry)
        else:
            issue = {"detail": str(entry)}
        issue.setdefault("check", "external_review")
        issue.setdefault("severity", "warn")
        issue.setdefault("kind", "external_issue")
        issue.setdefault("source", "external")
        issues.append(issue)
    return {"status": "ok", "command": command, "pass": bool(payload.get("pass", not issues)),
            "score": payload.get("score"), "summary": payload.get("summary"),
            "issues": issues, "suggestions": payload.get("suggestions") or [],
            "exit_code": result.returncode, "seconds": round(time.time() - started, 2)}


def check_external(ctx: dict, external: dict) -> dict:
    issues = []
    if external.get("status") == "ok" and external.get("pass") is False:
        if not external["issues"]:
            issues.append({"check": "external_review", "severity": "fail", "kind": "external_reject",
                           "detail": "the external reviewer rejected the scene", "source": "external"})
        else:
            for issue in external["issues"]:
                if issue.get("severity", "warn") not in ("fail", "warn"):
                    issue["severity"] = "warn"
                issue.setdefault("source", "external")
            issues = external["issues"]
    elif external.get("status") == "ok":
        issues = [issue for issue in external["issues"] if issue.get("severity") == "warn"]
        for issue in issues:
            issue.setdefault("source", "external")
    details = {key: value for key, value in external.items() if key not in ("issues", "suggestions")}
    return _check("external_review", "render", "fail", issues, **details)


# --------------------------------------------------------------------------- #
# the review itself
# --------------------------------------------------------------------------- #
def review(outdir: str, spec_path: Optional[str] = None, manifest_path: Optional[str] = None,
           index: Optional[Dict[str, dict]] = None, images: bool = True, depth: bool = True,
           review_cmd: Optional[str] = None, strict: bool = False, spec_only: bool = False,
           timeout: float = 180.0) -> dict:
    """Run every check that has the artifacts for it and return the report dict."""
    started = time.time()
    artifacts = Artifacts(outdir, spec_path, manifest_path)
    if index is None:
        index = checks_mod.load_assets_index()
    report: dict = {
        "kind": "blender_agent_review",
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "scene": artifacts.outdir,
        "spec": artifacts.spec_path,
        "manifest": artifacts.manifest_path if artifacts.has_build else None,
        "strict": bool(strict),
    }
    if not artifacts.has_spec:
        report.update({"pass": False, "score": 0.0, "scopes": {"spec": False, "build": False,
                                                              "render": False},
                       "checks": [], "issues": [{"check": "spec", "severity": "fail",
                                                 "kind": "missing_spec",
                                                 "detail": f"no spec at {artifacts.spec_path}",
                                                 "repair": {"kind": "note"}}],
                       "error": "missing_spec", "seconds": round(time.time() - started, 3)})
        return report

    spec = checks_mod.spec_mod.load_spec(artifacts.spec_path)
    boxes, unknown = checks_mod.item_boxes(spec, index)
    items_by_room: Dict[str, List[checks_mod.ItemBox]] = {}
    for box in boxes:
        items_by_room.setdefault(box.room, []).append(box)
    ctx = {"artifacts": artifacts, "spec": spec, "index": index, "boxes": boxes,
           "items_by_room": items_by_room, "depth": bool(depth)}

    checks = [
        check_rooms_furnished(ctx),
        check_functional_coverage(ctx),
        check_no_duplicate_furniture(ctx),
        check_camera_clearance(ctx),
        check_camera_framing(ctx),
        check_camera_coverage(ctx),
    ]
    if unknown:
        checks.append(_check("assets_resolved", "spec", "warn",
                             [{"check": "assets_resolved", "severity": "warn", "kind": "unknown_asset",
                               "detail": note, "repair": {"kind": "note"}} for note in unknown],
                             unknown=len(unknown)))

    external = {"status": "not_configured", "command": None, "issues": []}
    if not spec_only:
        if artifacts.has_build:
            checks.append(check_build_matches_spec(ctx))
            checks.append(check_furniture_grounding(ctx))
        else:
            checks.append(_empty("not_run", "build_matches_spec", "build", "fail",
                                 "no build/build_manifest.json"))
            checks.append(_empty("not_run", "furniture_grounding", "build", "fail",
                                 "no build/build_manifest.json"))
        if images:
            checks.append(check_exposure(ctx))
        else:
            checks.append(_empty("not_run", "exposure", "render", "fail", "images disabled"))
        external = run_external_review(ctx, review_cmd, timeout=timeout)
        checks.append(check_external(ctx, external))

    issues: List[dict] = [issue for check in checks for issue in check["issues"]]
    fails = [issue for issue in issues if issue.get("severity", "warn") == "fail"]
    warns = [issue for issue in issues if issue.get("severity", "warn") == "warn"]
    report.update({
        "scopes": {"spec": True, "build": bool(artifacts.has_build) and not spec_only,
                   "render": bool(artifacts.panoramas()) and not spec_only and images},
        "pass": bool(not issues if strict else not fails),
        "score": round(max(0.0, 1.0 - 0.2 * len(fails) - 0.05 * len(warns)), 3),
        "checks": checks,
        "issues": issues,
        "suggestions": [issue["repair"] for issue in issues if issue.get("repair")
                        and issue["repair"].get("kind") not in (None, "note")],
        "external": {key: value for key, value in external.items() if key != "issues"},
        "stats": {"rooms": len(spec.rooms), "furniture": len(spec.furniture),
                  "cameras": len(spec.cameras), "openings": len(spec.openings),
                  "windows": len(spec.windows), "panoramas": len(artifacts.panoramas()),
                  "issues_fail": len(fails), "issues_warn": len(warns)},
        "seconds": round(time.time() - started, 3),
    })
    if external.get("issues"):
        report["external"]["issues"] = external["issues"]
    return report


def render_text(report: dict) -> str:
    """One-screen summary for the shell hooks and the console."""
    stats = report.get("stats") or {}
    lines = [f"review: {'PASS' if report.get('pass') else 'FAIL'} "
             f"(score {report.get('score')}, {stats.get('issues_fail', 0)} fail / "
             f"{stats.get('issues_warn', 0)} warn)"]
    for check in report.get("checks", []):
        if check.get("status") == "not_run":
            lines.append(f"  {check['name']:22s} not run ({check['details'].get('reason')})")
        elif not check["pass"]:
            lines.append(f"  {check['name']:22s} {check['issues_fail']} fail / "
                         f"{check['issues_warn']} warn")
    for issue in report.get("issues", [])[:12]:
        lines.append(f"  [{issue.get('severity', 'warn')}] {issue.get('check')}: {issue.get('detail')}")
    if len(report.get("issues", [])) > 12:
        lines.append(f"  ... {len(report['issues']) - 12} more issue(s) in {DEFAULT_REPORT_NAME}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# repair: review suggestions -> spec tweaks
# --------------------------------------------------------------------------- #
def _free_for(items: Sequence[checks_mod.ItemBox], room_id: str, box,
              cameras: Sequence, doors: Sequence, cameras_gap: float = checks_mod.CAMERA_FURNITURE_GAP_M + 0.05,
              item_gap: float = 0.05) -> bool:
    """Cheap pre-filter before the authoritative `checks.py` run."""
    for other in items:
        if other.room != room_id:
            continue
        if checks_mod.rect_distance(box.aabb, other.aabb) < item_gap:
            return False
    for zone in doors:
        for room, rect in zone.rects.items():
            if room != room_id:
                continue
            if checks_mod.rect_distance(box.aabb, rect) < checks_mod.DOOR_CLEARANCE_M + 0.05:
                return False
    for camera in cameras:
        if camera.room != room_id:
            continue
        point = (camera.pos[0], camera.pos[1], camera.pos[0], camera.pos[1])
        if checks_mod.rect_distance(box.aabb, point) < cameras_gap:
            return False
    return True


def _wall_candidates(rect: Tuple[float, float, float, float], dims: Sequence[float],
                     step: float = 0.25, limit: int = 14) -> List[Tuple[List[float], float]]:
    """``(pos, rot_z_deg)`` wall-hug candidates, longest wall first.

    Uses the wall table of ``layout_gen``: side "S" is the wall at ``y0`` and
    faces +Y, "N" the one at ``y1`` facing -Y, "W"/"E" the x walls.  A piece
    stands ``wall/2 + 2 cm`` off the wall line (``BACK_GAP_M``).
    """
    x0, y0, x1, y1 = rect
    back = 0.06 + 0.02                                   # wall half thickness + gap
    # ``along`` runs parallel to the wall, ``across`` perpendicular to it -- and
    # the yaw turns the piece, so the x walls take dims[0]/dims[1] in the other
    # order (rot 90 maps the piece's local +X onto +Y)
    walls = [
        ("S", y0, x0, x1, dims[0], dims[1], 180.0, (0.0, 1.0)),
        ("N", y1, x0, x1, dims[0], dims[1], 0.0, (0.0, -1.0)),
        ("W", x0, y0, y1, dims[0], dims[1], 90.0, (1.0, 0.0)),
        ("E", x1, y0, y1, dims[0], dims[1], 270.0, (-1.0, 0.0)),
    ]
    walls.sort(key=lambda wall: -(wall[3] - wall[2]))
    out: List[Tuple[List[float], float]] = []
    for side, line, lo, hi, along, across, rot, facing in walls:
        span = hi - lo - along
        if span < 0.0:
            continue
        mid = lo + span / 2.0
        offsets: List[float] = [0.0]
        for k in range(1, limit):
            offsets.append((k if k % 2 else -k) * step)
        for offset in offsets:
            centre = mid + offset
            if not (lo - 1e-9 <= centre - along / 2.0 and centre + along / 2.0 <= hi + 1e-9):
                continue
            off = back + across / 2.0
            if side in ("S", "N"):
                pos = [centre, line + facing[1] * off]
            else:
                pos = [line + facing[0] * off, centre]
            out.append((pos, rot))
            if len(out) >= limit:
                return out
    return out


def _usage_free(items: Sequence[checks_mod.ItemBox], room_id: str, pos: Sequence[float],
                dims: Sequence[float], rot: float, asset: str) -> bool:
    """Keep the clear space a fixture's use needs (fridge door, pulled-out chair)."""
    if not asset.startswith("builtin:"):
        return True
    part = fixtures_mod.FIXTURE_PARTS.get(asset.split(":", 1)[1])
    if part is None or part.get("usage_m", 0.0) <= 0.0:
        return True
    radians = math.radians(rot)
    facing = (math.sin(radians), -math.cos(radians))     # rot 0 faces -Y, 90 faces +X
    half_x = abs(math.cos(radians)) * dims[0] / 2.0 + abs(math.sin(radians)) * dims[1] / 2.0
    half_y = abs(math.sin(radians)) * dims[0] / 2.0 + abs(math.cos(radians)) * dims[1] / 2.0
    usage = float(part["usage_m"])
    if abs(facing[0]) > abs(facing[1]):
        cx = pos[0] + facing[0] * (half_x + usage / 2.0)
        zone = (cx - usage / 2.0, pos[1] - half_y, cx + usage / 2.0, pos[1] + half_y)
    else:
        cy = pos[1] + facing[1] * (half_y + usage / 2.0)
        zone = (pos[0] - half_x, cy - usage / 2.0, pos[0] + half_x, cy + usage / 2.0)
    for other in items:
        if other.room != room_id:
            continue
        if checks_mod.rect_distance(zone, other.aabb) < checks_mod.WALL_GAP_MIN_M:
            return False
    return True


def _place_furniture(raw: dict, spec, boxes: Sequence[checks_mod.ItemBox], room, asset: str,
                     index: Dict[str, dict]) -> Optional[Tuple[dict, dict]]:
    """``(placement, patched spec json)`` for ``asset`` against a wall of ``room``."""
    dims = checks_mod.item_dims(asset, index)
    rect = _rect_of_polygon(room.polygon)
    if dims is None or rect is None:
        return None
    doors = checks_mod.door_zones(spec, spec.wall_thickness)
    for pos, rot in _wall_candidates(rect, dims):
        aabb = checks_mod.rotated_aabb(pos, dims, rot)
        if checks_mod.wall_gap(aabb, room.polygon) < checks_mod.WALL_GAP_MIN_M:
            continue
        probe = checks_mod.ItemBox(name="__repair__", asset=asset, room=room.id, aabb=aabb,
                                   z=(0.0, dims[2]), dims=tuple(dims), placement=checks_mod.WALL_HUG,
                                   rot_z_deg=rot, rotation_on_grid=True)
        if not _free_for(boxes, room.id, probe, spec.cameras, doors):
            continue
        if not _usage_free(boxes, room.id, pos, dims, rot, asset):
            continue
        entry = {"asset": asset, "room": room.id, "pos": [round(pos[0], 3), round(pos[1], 3)],
                 "rot_z_deg": rot, "name": _repair_name(asset), "placement": "wall_hug"}
        trial = json.loads(json.dumps(raw))
        trial.setdefault("furniture", []).append(entry)
        patched = _load_from_dict(trial)
        if patched is None:
            return None
        result = checks_mod.check_spec(patched, index)
        if result["pass"]:
            return entry, trial
    return None


def _repair_name(asset: str) -> str:
    stem = asset.split(":", 1)[-1].replace("-", "_")
    return f"review_{stem}"


def _load_from_dict(payload: dict):
    import tempfile

    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    try:
        json.dump(payload, handle)
        handle.close()
        return checks_mod.spec_mod.load_spec(handle.name)
    finally:
        os.unlink(handle.name)


def _camera_candidates(spec, boxes: Sequence[checks_mod.ItemBox], camera, index, step: float = 0.25):
    """Free stands for one camera, best first (furniture in front, off the walls)."""
    room = next((room for room in spec.rooms if room.id == camera.room), None)
    rect = _rect_of_polygon(room.polygon) if room is not None else None
    if rect is None:
        return []
    x0, y0, x1, y1 = rect
    items = [box for box in boxes if box.room == camera.room and not _is_decor(box.asset, index)]
    doors = checks_mod.door_zones(spec, spec.wall_thickness)
    door_rects = [zone.rects[camera.room] for zone in doors if camera.room in zone.rects]
    out = []
    x = x0 + step
    while x <= x1 - step + 1e-9:
        y = y0 + step
        while y <= y1 - step + 1e-9:
            point = (x, y, x, y)
            gap = checks_mod.wall_gap(point, room.polygon)
            nearest = min([checks_mod.rect_distance(point, box.aabb) for box in items],
                          default=float("inf"))
            door_gap = min([checks_mod.rect_distance(point, rect) for rect in door_rects],
                           default=float("inf"))
            if gap >= WALL_HUG_WARN_M and nearest >= checks_mod.CAMERA_FURNITURE_GAP_M + 0.05 \
                    and door_gap >= checks_mod.DOOR_CLEARANCE_M + 0.05:
                framing = camera_framing(items, [x, y, 1.5])
                score = (framing["volume_share"] * 1.5 + min(gap, 1.4) / 1.4 * 0.7
                         + min(nearest, 1.4) / 1.4 * 0.3 + min(door_gap, 1.5) / 1.5 * 0.2)
                out.append((score, [round(x, 3), round(y, 3), camera.pos[2]],
                            framing["volume_share"], round(gap, 3)))
            y += step
        x += step
    out.sort(key=lambda entry: -entry[0])
    return out


def apply_repair(outdir: str, report: Optional[dict] = None, index: Optional[Dict[str, dict]] = None,
                 spec_path: Optional[str] = None, dry_run: bool = False,
                 max_actions: int = 6, report_path: Optional[str] = None) -> dict:
    """Turn the report's repair hints into spec tweaks, validated by `checks.py`.

    Only actions that keep the *whole* spec passing ``checks.check_spec`` are
    kept, one at a time, so a repair can never publish a broken layout.  The
    patched spec is written to ``spec.json`` (the previous file is kept as
    ``spec.pre_review.json``) and described in ``review_repair.json``.
    """
    artifacts = Artifacts(outdir, spec_path)
    if report is None:
        report = _load_json(report_path or os.path.join(artifacts.outdir, DEFAULT_REPORT_NAME)) or {}
    if not artifacts.has_spec:
        return {"kind": "blender_agent_review_repair", "changed": False, "status": "error",
                "error": f"no spec at {artifacts.spec_path}", "actions": []}
    if index is None:
        index = checks_mod.load_assets_index()
    raw = json.loads(json.dumps(artifacts.raw_spec))
    spec = checks_mod.spec_mod.load_spec(artifacts.spec_path)
    boxes, _unknown = checks_mod.item_boxes(spec, index)
    rooms = {room.id: room for room in spec.rooms}

    actions: List[dict] = []
    skipped: List[dict] = []

    def try_spec(payload: dict) -> Tuple[bool, Optional[dict]]:
        candidate = _load_from_dict(payload)
        if candidate is None:
            return False, None
        result = checks_mod.check_spec(candidate, index)
        return bool(result["pass"]), result

    # -- add furniture to a room that misses something --------------------- #
    wanted: Dict[str, str] = {}
    for issue in report.get("issues", []):
        repair = issue.get("repair") or {}
        if repair.get("kind") != "add_furniture":
            continue
        room_id = repair.get("room")
        if room_id in wanted or room_id not in rooms:
            continue
        room_type = repair.get("room_type") or rooms[room_id].type
        asset = _repair_asset(room_type, index)
        if asset:
            wanted[room_id] = asset
        else:
            skipped.append({"kind": "add_furniture", "room": room_id,
                            "why": f"no asset for room type '{room_type}'"})
    for room_id, asset in wanted.items():
        if len(actions) >= max_actions:
            break
        placement = _place_furniture(raw, spec, boxes, rooms[room_id], asset, index)
        if placement is None:
            fallback = REPAIR_FIXTURE.get(rooms[room_id].type)
            if fallback and fallback != asset:
                placement = _place_furniture(raw, spec, boxes, rooms[room_id], fallback, index)
                asset = fallback
        if placement is None:
            skipped.append({"kind": "add_furniture", "room": room_id, "asset": asset,
                            "why": "no free wall spot that keeps checks.py passing"})
            continue
        entry, trial = placement
        raw = trial
        spec = _load_from_dict(raw)
        boxes, _unknown = checks_mod.item_boxes(spec, index)
        actions.append(dict(entry, kind="add_furniture", why="room was empty or missing its function"))

    # -- move a camera off the wall and in front of the room ---------------- #
    moved: set = set()
    for issue in report.get("issues", []):
        repair = issue.get("repair") or {}
        if repair.get("kind") != "move_camera" or len(actions) >= max_actions:
            continue
        # the loop is about publication quality: move a stand that is a real
        # problem (or faces nothing at all), leave the soft warnings alone
        if issue.get("severity") != "fail" and issue.get("kind") != "subject_behind_camera":
            continue
        camera_name = repair.get("camera")
        if camera_name in moved:
            continue
        camera = next((entry for entry in spec.cameras if entry.name == camera_name), None)
        if camera is None:
            continue
        current = camera_framing([box for box in boxes if box.room == camera.room], camera.pos)
        room = next((entry for entry in spec.rooms if entry.id == camera.room), None)
        wall_before = checks_mod.wall_gap((camera.pos[0], camera.pos[1], camera.pos[0], camera.pos[1]),
                                          room.polygon) if room is not None else 0.0
        improvement = False
        for _score, pos, share, wall_after in _camera_candidates(spec, boxes, camera, index)[:8]:
            trial = json.loads(json.dumps(raw))
            for entry in trial["cameras"]:
                if entry["name"] == camera_name:
                    entry["pos"] = pos
            keep, _result = try_spec(trial)
            if not keep:
                continue
            better_share = share > current["volume_share"] + 0.02
            better_wall = wall_after > wall_before + 0.1
            if issue["kind"] == "subject_behind_camera" and not better_share:
                continue
            if issue["kind"] != "subject_behind_camera" and not (better_share or better_wall):
                continue
            raw = trial
            spec = _load_from_dict(raw)
            boxes, _unknown = checks_mod.item_boxes(spec, index)
            actions.append({"kind": "move_camera", "camera": camera_name, "room": camera.room,
                            "pos": pos, "why": issue.get("kind"),
                            "forward_share_before": current["volume_share"],
                            "forward_share_after": round(share, 3),
                            "wall_gap_before": round(wall_before, 3), "wall_gap_after": wall_after})
            moved.add(camera_name)
            improvement = True
            break
        if not improvement:
            skipped.append({"kind": "move_camera", "camera": camera_name,
                            "why": "no better stand kept checks.py passing"})

    # -- calm an over-lit style -------------------------------------------- #
    if any((issue.get("repair") or {}).get("kind") == "reduce_exposure"
           and issue.get("severity") == "fail" for issue in report.get("issues", [])):
        if len(actions) < max_actions:
            style = dict(raw.get("style") or {})
            exposure = float(style.get("exposure", 1.4))
            style["exposure"] = round(max(0.0, exposure - 0.3), 3)
            if raw.get("style"):
                raw["style"] = style
                spec = _load_from_dict(raw)
                actions.append({"kind": "reduce_exposure", "exposure_before": exposure,
                                "exposure_after": style["exposure"],
                                "why": "a panorama was blown out"})

    changed = bool(actions)
    result = {"kind": "blender_agent_review_repair", "schema_version": SCHEMA_VERSION,
              "created_utc": _utc_now(), "scene": artifacts.outdir,
              "source_report": os.path.join(artifacts.outdir, DEFAULT_REPORT_NAME),
              "changed": changed, "status": "applied" if changed else "nothing_to_do",
              "actions": actions, "skipped": skipped,
              "checks": {"pass": None}, "dry_run": bool(dry_run)}
    if changed:
        keep, checks_result = try_spec(raw)
        result["checks"] = {"pass": keep,
                            "failed": [check["name"] for check in (checks_result or {}).get("checks", [])
                                       if not check["pass"]]}
        if not keep:
            result.update({"changed": False, "status": "rejected",
                           "error": "the repaired spec failed checks.py"})
            return result
        if not dry_run:
            backup = os.path.join(artifacts.outdir, "spec.pre_review.json")
            if not os.path.isfile(backup):
                with open(backup, "w", encoding="utf-8") as handle:
                    json.dump(artifacts.raw_spec, handle, indent=2)
            with open(artifacts.spec_path, "w", encoding="utf-8") as handle:
                json.dump(raw, handle, indent=2)
            with open(os.path.join(artifacts.outdir, "review_repair.json"), "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
    return result


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Low-cost visual / rule review of a scene package")
    parser.add_argument("outdir", help="scene directory (spec.json, build/, scene/viewpoints/)")
    parser.add_argument("--spec", default=None, help="spec to review (default <outdir>/spec.json)")
    parser.add_argument("--manifest", default=None, help="build manifest (default <outdir>/build/build_manifest.json)")
    parser.add_argument("--json", dest="json_out", default=None,
                        help=f"report path (default <outdir>/{DEFAULT_REPORT_NAME})")
    parser.add_argument("--spec-only", action="store_true", help="skip the build and render checks")
    parser.add_argument("--no-images", action="store_true", help="skip the panorama pixel checks")
    parser.add_argument("--no-depth", action="store_true", help="skip the depth-map reads")
    parser.add_argument("--strict", action="store_true", help="warnings fail the review too")
    parser.add_argument("--review-cmd", default=os.environ.get("REVIEW_CMD"),
                        help="external reviewer (default: $REVIEW_CMD)")
    parser.add_argument("--timeout", type=float, default=180.0, help="external reviewer timeout (s)")
    parser.add_argument("--apply-repair", action="store_true",
                        help="patch spec.json from review_report.json (exit 0 when it changed)")
    parser.add_argument("--report", dest="report_in", default=None,
                        help="with --apply-repair: the report to take hints from "
                             "(default <outdir>/review_report.json)")
    parser.add_argument("--dry-run", action="store_true", help="with --apply-repair: do not write")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    outdir = os.path.abspath(args.outdir)
    if not os.path.isdir(outdir):
        print(f"review: no such directory: {outdir}", file=sys.stderr)
        return 2

    if args.apply_repair:
        try:
            result = apply_repair(outdir, spec_path=args.spec, dry_run=args.dry_run,
                                  report_path=args.report_in)
        except Exception as error:                       # never crash a batch on a repair
            import traceback

            result = {"kind": "blender_agent_review_repair", "changed": False, "status": "error",
                      "actions": [], "error": f"{type(error).__name__}: {error}",
                      "traceback": traceback.format_exc()[-2000:]}
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
        if not args.quiet:
            print(f"review repair: {result['status']} ({len(result.get('actions', []))} action(s))")
            for action in result.get("actions", []):
                print(f"  {action['kind']}: {json.dumps({k: v for k, v in action.items() if k != 'kind'})}")
            for entry in result.get("skipped", []):
                print(f"  skipped {entry.get('kind')}: {entry.get('why')}")
            if result.get("error"):
                print(f"  error: {result['error']}")
        return 0 if result.get("changed") else (2 if result["status"] == "error" else 1)

    try:
        report = review(outdir, spec_path=args.spec, manifest_path=args.manifest,
                        images=not args.no_images, depth=not args.no_depth,
                        review_cmd=args.review_cmd, strict=args.strict,
                        spec_only=args.spec_only, timeout=args.timeout)
    except Exception as error:                           # a review crash must be legible
        import traceback

        report = {"kind": "blender_agent_review", "schema_version": SCHEMA_VERSION,
                  "created_utc": _utc_now(), "scene": outdir, "pass": False, "score": 0.0,
                  "error": "internal_error", "error_detail": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()[-2000:], "checks": [], "issues": [],
                  "stats": {"issues_fail": 0, "issues_warn": 0}}
    json_out = args.json_out or os.path.join(outdir, DEFAULT_REPORT_NAME)
    try:
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    except OSError as error:
        print(f"review: cannot write {json_out}: {error}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(render_text(report))
    if report.get("error") in ("missing_spec", "internal_error"):
        return 2
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
