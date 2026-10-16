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
``"severity": "fail"`` fail a scene, the rest are warnings that ride along in
the report.  ``--strict`` promotes warnings to failures.  The gates are the
computed defects a downstream training set cannot use (an empty habitable room,
a kitchen without a kitchen, furniture that settled through the floor, a build
made from a different spec, an interior wall blown out into a white blob); the
warnings are the quality headroom (a camera in the comfort band along a wall, a
viewpoint facing the empty side of its room, a bright window that clips a large
part of the frame).

Four states, not two
--------------------
A boolean ``pass`` cannot express "nobody looked".  Every check reports a
``status``:

``ok`` / ``issues``
    the check ran; ``pass`` is true only when it found no issue at all
``not_run``
    the artifact the check reads is missing (a full review of a directory
    without ``build/build_manifest.json`` or without panoramas), or the caller
    asked for the check to be skipped.  A *required* ``not_run`` check is **not
    a pass**: the report carries ``status: "not_run"``, the names of the
    missing checks in ``missing`` and ``pass: false``.
``error``
    the check could not produce a verdict (an external reviewer that crashed or
    printed junk).  It is retryable, never silently green: the report is
    ``status: "error"`` unless the caller opts into degrading it.

``required`` is recorded per check and as ``scopes_required`` on the report
(``spec`` always, ``build``/``render`` for a full review, ``--spec-only``
requires the spec alone, ``--no-images`` drops the render scope); a check the
caller explicitly skipped is ``not_run`` with ``required: false`` and does not
block.  An external reviewer that answers ``pass: false`` always fails the
scene, its own fail-level issues are never dropped, and one that cannot be
reached is retryable -- degrade it to a warning only with
``--allow-external-error`` / ``PANOWORLD_REVIEW_ALLOW_EXTERNAL_ERROR=1``.

Command line::

    python blender_agent/review.py <outdir> [--json review_report.json]
    python blender_agent/review.py <outdir> --spec-only        # no build/render reads
    python blender_agent/review.py <outdir> --apply-repair     # patch spec.json
    python blender_agent/review.py <outdir> --fix-exposure     # redo the colour management

Exit codes: ``0`` pass, ``1`` review failed / not run / error, ``2`` usage/IO
error; for ``--apply-repair``: ``0`` spec changed, ``1`` nothing to change,
``2`` error; ``--fix-exposure``: ``0`` the panoramas changed, ``1`` nothing to
change, ``2`` error.

External reviewers plug in through the environment: ``REVIEW_CMD`` is split
with :func:`shlex.split` and gets the panorama image paths appended; it must
print one JSON object on stdout (``{"pass": bool, "issues": [...],
"suggestions": [...]}``) and is run with a timeout.

Exposure without a re-render
----------------------------
``--fix-exposure`` reads the *linear* EXRs the build keeps next to the
panoramas, simulates the scene's display transform at candidate exposures
(equirectangular pixels weighted by cos(latitude), clipped pixels split into
interior surfaces and the sky/window behind them), picks **one** exposure for
the whole viewpoint set and redoes the colour management of the existing render
-- denoise and display transform, no modelling, no ray tracing -- then
re-reviews the images it wrote.  See :func:`fix_exposure`.
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
SCHEMA_VERSION = 2

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
# The gate is on the *interior* clipped share: a bright window or the sky
# behind it clips by design and is quality headroom, a wall or a floor that
# turns into a white sheet is not.  Both are measured on the equirectangular
# frame with a cos(latitude) pixel weight, because the rows near a pole cover a
# vanishing solid angle and a metre of blown ceiling there is not a metre of
# blown wall at eye level.
CLIP_LEVEL = 0.98
CLIP_WARN_FRACTION = 0.15
CLIP_FAIL_FRACTION = 0.35
CLIP_BLOB_WARN = 0.08
CLIP_BLOB_FAIL = 0.25
DARK_WARN_LUMINANCE = 0.08
# A hit farther than this is the sky dome / outside, not a room surface.  The
# dome radius comes from the manifest, so a bigger shell keeps working.
EXTERIOR_DEPTH_FRACTION = 0.5
EXTERIOR_DEPTH_FLOOR_M = 10.0
# The exposure re-do (`--fix-exposure`): the search runs over whole stops in
# this range, a stop in EXPOSURE_STEP sub-steps; FIX_MARGIN keeps the target
# clear of the threshold instead of landing exactly on it.
EXPOSURE_STEP = 0.05
EXPOSURE_FLOOR_STOPS = -4.0
FIX_MARGIN = 0.8
FIX_MAX_ROUNDS = 3

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

    @property
    def exterior_depth_m(self) -> float:
        """Distance beyond which a depth hit is the sky dome, not a room surface."""
        radius = ((self.manifest or {}).get("shell") or {}).get("dome_radius")
        try:
            radius = float(radius)
        except (TypeError, ValueError):
            radius = 0.0
        return max(EXTERIOR_DEPTH_FLOOR_M, EXTERIOR_DEPTH_FRACTION * radius)

    def raw_exrs(self) -> Dict[str, str]:
        """``camera name -> linear EXR`` the build kept (``--no-denoise`` flow)."""
        build = os.path.join(self.outdir, "build")
        out: Dict[str, str] = {}
        for entry in (self.manifest or {}).get("cameras") or []:
            name, raw = entry.get("camera"), entry.get("raw_exr")
            if name and raw and os.path.isfile(os.path.join(build, raw)):
                out[name] = os.path.join(build, raw)
        if out:
            return out
        if os.path.isdir(build):
            for entry in sorted(os.listdir(build)):
                if entry.startswith("raw_") and entry.endswith(".exr"):
                    stem = entry[len("raw_"):-len(".exr")]
                    if len(stem) > 4 and stem[-4:].isdigit():   # raw_<camera><frame>.exr
                        stem = stem[:-4]
                    out[stem] = os.path.join(build, entry)
        return out

    def viewpoint_pano(self, camera: str) -> Optional[str]:
        view = os.path.join(self.viewpoints, camera)
        if not os.path.isdir(view):
            return None
        for entry in sorted(os.listdir(view)):
            if entry.startswith("panoImage_") and entry.endswith(".png"):
                return os.path.join(view, entry)
        return None

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
           required: bool = True, **details: object) -> dict:
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
        "required": bool(required),
        "issues": issues,
        "issues_fail": len(fails),
        "issues_warn": len(issues) - len(fails),
        "details": details,
    }


def _empty(status: str, name: str, scope: str, severity: str, reason: str,
           required: bool = False, passed: Optional[bool] = None) -> dict:
    """A check that did not run (`not_run`) or could not run (`error`).

    ``required`` is what decides whether the *report* can still pass: a
    required check without a verdict is not a pass, a check the caller
    explicitly skipped is.
    """
    ok = (not required) if passed is None else bool(passed)
    return {"name": name, "scope": scope, "severity": severity, "gate": severity,
            "pass": ok, "status": status, "required": bool(required), "issues": [],
            "issues_fail": 0, "issues_warn": 0, "details": {"reason": reason}}


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
                      "the manifest has no settle data (built before settle.py, or --skip-render)",
                      required=bool(ctx.get("build_required", True)))
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
def _erp_weights(height: int):
    """``cos(latitude)`` pixel weight of an equirectangular frame, mean 1.

    Row 0 is the top pole (weight 0), the middle row the horizon (weight 1):
    the polar rows cover a vanishing solid angle, so a blown ceiling strip in
    them must not count like the same strip of wall at eye level.
    """
    import numpy as np

    theta = (np.arange(height, dtype=np.float64) + 0.5) / float(height) * math.pi
    weights = np.sin(theta)
    return weights / weights.mean()


def _weighted_share(mask, weights) -> float:
    """Share of ``mask`` under the per-row ERP weights."""
    import numpy as np

    total = float(np.asarray(weights).sum())
    if total <= 0.0:
        return float(mask.mean()) if mask.size else 0.0
    return float((mask * np.asarray(weights)[:, None]).sum() / total / mask.shape[1])


def _clip_split(clipped, depth_path: Optional[str], exterior_depth_m: float):
    """``(interior, exterior)`` clipped masks, split by the depth map.

    A clipped pixel whose ray runs past the shell (the sky dome, or a hit
    outside the depth range) is exterior; without a depth map every clipped
    pixel counts as interior, which is the conservative reading.  Note that a
    *glazed* window records the glass, not the sky, so it lands in the interior
    bucket -- the split catches open openings and the dome above the walls.
    """
    import numpy as np

    if not depth_path:
        return clipped, np.zeros_like(clipped)
    try:
        depth = np.load(depth_path).astype("float32")
    except Exception:
        return clipped, np.zeros_like(clipped)
    if depth.shape != clipped.shape:                 # the ERP was measured downsampled
        factor = max(1, int(round(depth.shape[0] / float(clipped.shape[0]))))
        if depth.shape[0] != clipped.shape[0] * factor \
                or depth.shape[1] != clipped.shape[1] * factor:
            return clipped, np.zeros_like(clipped)
        depth = _downsample(depth, factor)
    exterior = ~np.isfinite(depth) | (depth <= 0.0) | (depth >= exterior_depth_m)
    return clipped & ~exterior, clipped & exterior


def _pano_stats(path: str) -> Optional[Tuple[dict, object]]:
    """Clipped-pixel fraction, largest clipped blob and luminance of one ERP png.

    Returns ``(stats, clipped_mask)``; the mask is the erp-resolution
    ``luminance >= CLIP_LEVEL`` boolean image the caller splits into interior
    and exterior pixels.
    """
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
    stats = {
        "size": [int(pixels.shape[1]), int(pixels.shape[0])],
        "clip_fraction": float(clipped.mean()),
        "largest_blob_fraction": _largest_blob_fraction(clipped),
        "mean_luminance": float(luminance.mean()),
        "p99_luminance": float(np.percentile(luminance, 99.0)),
    }
    return stats, clipped


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
    """Blown highlights and dark viewpoints in the rendered panoramas.

    The gate is the share of *interior* pixels at the clip level: a white wall,
    a floor or a ceiling that has lost its texture is a defect, while the sky
    seen through a window (and the dome behind it) is allowed to clip -- it is
    reported as a ``clipped_window`` warning at most.  Interior and exterior are
    told apart by the depth map: a hit farther than
    :attr:`Artifacts.exterior_depth_m` is outside the shell.  Without depth maps
    nothing can be split, so every clipped pixel counts as interior and the
    plain frame fraction decides, exactly as before.

    Fractions are measured with an equirectangular ``cos(latitude)`` pixel
    weight, so a blown ceiling strip above the camera does not weigh as much as
    the same strip of wall at eye level.
    """
    artifacts = ctx["artifacts"]
    panoramas = artifacts.panoramas()
    required = bool(ctx.get("render_required", True))
    if not panoramas:
        return _empty("not_run", "exposure", "render", "fail", "no exported panoramas",
                      required=required)
    depths = artifacts.depth_maps() if ctx.get("depth", True) else {}
    exterior_depth = artifacts.exterior_depth_m
    issues: List[dict] = []
    per_camera: Dict[str, dict] = {}
    for camera, path in panoramas:
        measured = _pano_stats(path)
        if measured is None:
            continue
        stats, clipped = measured
        record = dict(stats)
        if camera in depths:
            depth = _depth_stats(depths[camera])
            if depth:
                record.update(depth)
        interior, exterior = _clip_split(clipped, depths.get(camera), exterior_depth)
        weights = _erp_weights(int(clipped.shape[0]))
        interior_share = _weighted_share(interior, weights)
        exterior_share = _weighted_share(exterior, weights)
        interior_blob = _largest_blob_fraction(interior)
        exterior_blob = _largest_blob_fraction(exterior)
        record.update({
            "clip_fraction_erp": round(_weighted_share(clipped, weights), 4),
            "interior_clip_fraction": round(interior_share, 4),
            "exterior_clip_fraction": round(exterior_share, 4),
            "interior_blob_fraction": round(interior_blob, 4),
            "exterior_blob_fraction": round(exterior_blob, 4),
            # the two are only apart when a depth map could be read
            "clipping": ("unclassified" if camera not in depths else
                         ("interior+exterior" if exterior_share > 0 else "interior-only")),
        })
        per_camera[camera] = {key: (round(value, 4) if isinstance(value, float) else value)
                              for key, value in record.items()}
        reason = None
        if interior_share >= CLIP_FAIL_FRACTION or interior_blob >= CLIP_BLOB_FAIL:
            severity, kind = "fail", "blown_out_view"
            reason = (f"{interior_share * 100:.1f} % of the frame is clipped indoors "
                      f"(largest blob {interior_blob * 100:.1f} %)")
        elif interior_share >= CLIP_WARN_FRACTION or interior_blob >= CLIP_BLOB_WARN:
            severity, kind = "warn", "bright_blob"
            reason = (f"{interior_share * 100:.1f} % of the frame is clipped indoors "
                      f"(largest blob {interior_blob * 100:.1f} %)")
        if reason:
            issues.append({"check": "exposure", "severity": severity, "kind": kind, "camera": camera,
                           "detail": f"camera '{camera}': {reason}",
                           "repair": {"kind": "reduce_exposure", "camera": camera}})
        if exterior_share >= CLIP_WARN_FRACTION or exterior_blob >= CLIP_BLOB_WARN:
            # a window, or the sky behind it: quality headroom, never a gate
            issues.append({"check": "exposure", "severity": "warn", "kind": "clipped_window",
                           "camera": camera,
                           "detail": f"camera '{camera}': {exterior_share * 100:.1f} % of the frame "
                                     f"clips on the sky/window side (largest blob "
                                     f"{exterior_blob * 100:.1f} %)",
                           "repair": {"kind": "note"}})
        if stats["mean_luminance"] < DARK_WARN_LUMINANCE:
            issues.append({"check": "exposure", "severity": "warn", "kind": "dark_view",
                           "camera": camera,
                           "detail": f"camera '{camera}': mean luminance "
                                     f"{stats['mean_luminance']:.3f} (viewpoint is very dark)",
                           "repair": {"kind": "add_light", "camera": camera}})
    return _check("exposure", "render", "fail", issues, required=required, cameras=per_camera)


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


def check_external(ctx: dict, external: dict, required: bool = True,
                   allow_error: bool = False) -> dict:
    """Fold the external verdict in -- without letting it evaporate.

    Three rules the boolean round of this check got wrong:

    * an explicit ``pass: false`` always fails, even when the reviewer only
      attached default (warn) notes -- an external rejection is a rejection;
    * every issue the reviewer reported is kept, a ``fail`` one especially; a
      reviewer that says ``pass: true`` while listing a fail issue is a
      contradiction the report shows rather than swallows;
    * a reviewer that cannot be reached (missing command, crash, junk on
      stdout) is ``status: "error"`` and blocks the pass, because "nobody
      looked" is not "it is fine".  ``allow_error`` degrades that to a warning
      for callers who run the external hook as an optional extra.
    """
    status = external.get("status") or "error"
    if status in ("not_configured", "skipped_no_images"):
        reason = ("no REVIEW_CMD configured" if status == "not_configured"
                  else "no exported panoramas to review")
        return _empty("not_run", "external_review", "render", "fail", reason, required=False)
    if status == "error":
        detail = f"the external reviewer did not answer: {external.get('error') or status}"
        check = _empty("error", "external_review", "render", "fail", detail,
                       required=required and not allow_error)
        check["details"].update({key: external[key] for key in
                                 ("command", "exit_code", "seconds", "stdout_tail", "stderr_tail")
                                 if external.get(key) is not None})
        check["details"]["retry"] = True
        if required and not allow_error:
            check["issues"] = [{"check": "external_review", "severity": "fail",
                                "kind": "external_unavailable", "source": "external",
                                "detail": detail,
                                "repair": {"kind": "retry_external_review"}}]
            check["issues_fail"] = 1
            check["severity"] = "fail"
        return check
    issues = []
    for entry in external.get("issues") or []:
        issue = dict(entry) if isinstance(entry, dict) else {"detail": str(entry)}
        if issue.get("severity") not in ("fail", "warn"):
            issue["severity"] = "warn"
        issue.setdefault("check", "external_review")
        issue.setdefault("kind", "external_issue")
        issue.setdefault("source", "external")
        issues.append(issue)
    fails = [issue for issue in issues if issue["severity"] == "fail"]
    verdict = external.get("pass")
    if verdict is False and not fails:
        issues.insert(0, {"check": "external_review", "severity": "fail",
                          "kind": "external_reject", "source": "external",
                          "detail": "the external reviewer rejected the scene",
                          "repair": {"kind": "note"}})
    details = {key: value for key, value in external.items() if key not in ("issues", "suggestions")}
    details["verdict_pass"] = verdict
    details["verdict_conflict"] = bool(verdict is True and fails)
    return _check("external_review", "render", "fail", issues, required=required, **details)


# --------------------------------------------------------------------------- #
# the review itself
# --------------------------------------------------------------------------- #
def review(outdir: str, spec_path: Optional[str] = None, manifest_path: Optional[str] = None,
           index: Optional[Dict[str, dict]] = None, images: bool = True, depth: bool = True,
           review_cmd: Optional[str] = None, strict: bool = False, spec_only: bool = False,
           timeout: float = 180.0, allow_external_error: bool = False,
           no_build: bool = False) -> dict:
    """Run every check that has the artifacts for it and return the report dict.

    ``pass`` is true only when every *required* check ran and found no failing
    issue; ``status`` says which of the four states the package is in
    (``pass``/``fail``/``not_run``/``error``) and ``missing`` names the
    required checks that had nothing to read.

    ``spec_only`` skips the build/render checks (the pre-render hook);
    ``no_build`` runs them but declares the build/render scope *not requested*
    (a layout-only run: ``PANOWORLD_SKIP_RENDER=1``), so an empty build
    directory is an explicit skip instead of a missing input.
    """
    started = time.time()
    artifacts = Artifacts(outdir, spec_path, manifest_path)
    if index is None:
        index = checks_mod.load_assets_index()
    optional_scope = spec_only or no_build
    required_scopes = {"spec": True, "build": not optional_scope,
                       "render": bool(images) and not optional_scope}
    external_required = bool(review_cmd) and not optional_scope
    report: dict = {
        "kind": "blender_agent_review",
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "scene": artifacts.outdir,
        "spec": artifacts.spec_path,
        "manifest": artifacts.manifest_path if artifacts.has_build else None,
        "strict": bool(strict),
        "scopes_required": dict(required_scopes),
    }
    if not artifacts.has_spec:
        report.update({"pass": False, "status": "error", "score": 0.0,
                       "scopes": {"spec": False, "build": False, "render": False},
                       "missing": ["spec"], "retry": False,
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
           "items_by_room": items_by_room, "depth": bool(depth),
           "build_required": required_scopes["build"], "render_required": required_scopes["render"]}

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
            absent = "the run asked for no build (layout-only review)" if no_build \
                else "no build/build_manifest.json"
            checks.append(_empty("not_run", "build_matches_spec", "build", "fail", absent,
                                 required=required_scopes["build"]))
            checks.append(_empty("not_run", "furniture_grounding", "build", "fail", absent,
                                 required=required_scopes["build"]))
        if images:
            checks.append(check_exposure(ctx))
        else:
            checks.append(_empty("not_run", "exposure", "render", "fail", "images disabled",
                                 required=False))
        external = run_external_review(ctx, review_cmd, timeout=timeout)
        checks.append(check_external(ctx, external, required=external_required,
                                     allow_error=allow_external_error))

    issues: List[dict] = [issue for check in checks for issue in check["issues"]]
    fails = [issue for issue in issues if issue.get("severity", "warn") == "fail"]
    warns = [issue for issue in issues if issue.get("severity", "warn") == "warn"]
    missing = [check["name"] for check in checks
               if check["status"] == "not_run" and check.get("required")]
    errored = [check["name"] for check in checks
               if check["status"] == "error" and not check.get("pass")]
    if errored:
        status = "error"
    elif missing:
        status = "not_run"
    elif fails or (strict and warns):
        status = "fail"
    else:
        status = "pass"
    external_block = {key: value for key, value in external.items() if key != "issues"}
    external_check = next((check for check in checks if check["name"] == "external_review"), None)
    if external_check is not None:
        external_block.update({key: value for key, value in external_check["details"].items()
                               if key != "reason"})
        external_block["required"] = external_check["required"]
        if external.get("issues"):
            external_block["issues"] = external["issues"]
    report.update({
        "scopes": {"spec": True, "build": bool(artifacts.has_build) and not spec_only,
                   "render": bool(artifacts.panoramas()) and not spec_only and images},
        "status": status,
        "pass": status == "pass",
        "missing": missing,
        "retry": bool(errored),
        "score": round(max(0.0, 1.0 - 0.2 * len(fails) - 0.05 * len(warns)), 3),
        "checks": checks,
        "issues": issues,
        "suggestions": [issue["repair"] for issue in issues if issue.get("repair")
                        and issue["repair"].get("kind") not in (None, "note")],
        "external": external_block,
        "stats": {"rooms": len(spec.rooms), "furniture": len(spec.furniture),
                  "cameras": len(spec.cameras), "openings": len(spec.openings),
                  "windows": len(spec.windows), "panoramas": len(artifacts.panoramas()),
                  "issues_fail": len(fails), "issues_warn": len(warns)},
        "seconds": round(time.time() - started, 3),
    })
    return report


def render_text(report: dict) -> str:
    """One-screen summary for the shell hooks and the console."""
    stats = report.get("stats") or {}
    state = (report.get("status") or ("pass" if report.get("pass") else "fail")).replace("_", " ")
    lines = [f"review: {state.upper()} "
             f"(score {report.get('score')}, {stats.get('issues_fail', 0)} fail / "
             f"{stats.get('issues_warn', 0)} warn)"]
    for name in report.get("missing") or []:
        lines.append(f"  missing input for required check: {name}")
    for check in report.get("checks", []):
        if check.get("status") == "not_run":
            lines.append(f"  {check['name']:22s} not run ({check['details'].get('reason')})")
        elif check.get("status") == "error":
            lines.append(f"  {check['name']:22s} error ({check['details'].get('reason')})")
        elif not check["pass"]:
            lines.append(f"  {check['name']:22s} {check['issues_fail']} fail / "
                         f"{check['issues_warn']} warn")
    for issue in report.get("issues", [])[:12]:
        lines.append(f"  [{issue.get('severity', 'warn')}] {issue.get('check')}: {issue.get('detail')}")
    if len(report.get("issues", [])) > 12:
        lines.append(f"  ... {len(report['issues']) - 12} more issue(s) in {DEFAULT_REPORT_NAME}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# exposure: linear HDR statistics and the colour-management re-do
# --------------------------------------------------------------------------- #
#: OpenEXR scan-line compressions this reader understands (the ones a Blender
#: compositor ``File Output`` writes without extra configuration).
_EXR_COMPRESSION = {0: "NONE", 1: "RLE", 2: "ZIPS", 3: "ZIP"}
_EXR_BLOCK_LINES = {0: 1, 1: 1, 2: 1, 3: 16}
_EXR_PIXEL_TYPES = {0: "<u4", 1: "<f2", 2: "<f4"}
#: the display transforms that can be reproduced without Blender
_PURE_PYTHON_TRANSFORMS = ("Standard", "Raw")


def _exr_channels(header: bytes) -> List[Tuple[str, int]]:
    """``(name, pixel type)`` of an OpenEXR ``chlist`` attribute."""
    out: List[Tuple[str, int]] = []
    index = 0
    while header[index:index + 1] != b"\x00":
        end = header.index(b"\x00", index)
        name = header[index:end].decode("ascii")
        index = end + 1
        pixel_type = int.from_bytes(header[index:index + 4], "little", signed=True)
        index += 8                       # pixel type + pLinear + 3 reserved bytes
        index += 8                       # x/y sampling
        out.append((name, pixel_type))
    return out


def _exr_unpredict(blob: bytes, transform: bool) -> bytes:
    """Undo the ZIP/RLE byte prediction (delta + even/odd interleave)."""
    import numpy as np

    raw = np.frombuffer(blob, dtype=np.uint8).astype(np.int32)
    if not transform:
        return raw.astype(np.uint8).tobytes()
    deltas = raw.copy()
    deltas[1:] -= 128
    recovered = np.cumsum(deltas) % 256
    half = (len(recovered) + 1) // 2
    out = np.empty(len(recovered), dtype=np.uint8)
    out[0::2] = recovered[:half][:len(out[0::2])]
    out[1::2] = recovered[half:][:len(out[1::2])]
    return out.tobytes()


def _rle_decode(payload: bytes) -> bytes:
    out = bytearray()
    index = 0
    while index < len(payload):
        count = int.from_bytes(payload[index:index + 1], "little", signed=True)
        index += 1
        if count < 0:
            out += payload[index:index - count]
            index += -count
        else:
            out += payload[index:index + 1] * (count + 1)
            index += 1
    return bytes(out)


def read_exr(path: str) -> dict:
    """Read an OpenEXR scan-line image into ``{name: (h, w) float32}``.

    Supports the uncompressed / RLE / ZIP / ZIPS variants a Blender render
    writes; PIZ, PXR24, B44 and DWA are refused with a legible error instead of
    a wrong image.  Only the pixels are decoded -- no chromaticities, no
    metadata -- because :func:`fix_exposure` only needs radiance.
    """
    import numpy as np

    with open(path, "rb") as handle:
        if handle.read(4) != b"\x76\x2f\x31\x01":
            raise RuntimeError(f"{path}: not an OpenEXR file")
        handle.read(4)                                    # version + flags
        attributes: Dict[str, Tuple[str, bytes]] = {}
        while True:
            name = b""
            while True:
                char = handle.read(1)
                if char in (b"", b"\x00"):
                    break
                name += char
            if not name:
                break
            kind = b""
            while True:
                char = handle.read(1)
                if char in (b"", b"\x00"):
                    break
                kind += char
            size = int.from_bytes(handle.read(4), "little", signed=True)
            attributes[name.decode("ascii")] = (kind.decode("ascii"), handle.read(size))
        compression = attributes["compression"][1][0]
        if compression not in _EXR_COMPRESSION:
            raise RuntimeError(f"{path}: EXR compression "
                               f"{attributes['compression'][1][0]} is not supported "
                               f"(NONE/RLE/ZIP/ZIPS only)")
        left, top, right, bottom = (int(value) for value in
                                    np.frombuffer(attributes["dataWindow"][1], dtype="<i4"))
        width, height = right - left + 1, bottom - top + 1
        channels = _exr_channels(attributes["channels"][1])
        block_lines = _EXR_BLOCK_LINES[compression]
        blocks = (height + block_lines - 1) // block_lines
        offsets = np.frombuffer(handle.read(8 * blocks), dtype="<u8")
        images = {name: np.zeros((height, width), dtype=np.float32) for name, _ in channels}
        for index, offset in enumerate(offsets):
            handle.seek(int(offset))
            handle.read(4)                                # y of the first line
            size = int.from_bytes(handle.read(4), "little", signed=True)
            payload = handle.read(size)
            lines = min(block_lines, height - index * block_lines)
            if compression == 0:
                raw = payload
            elif compression == 1:
                raw = _rle_decode(payload)
            else:
                import zlib
                try:
                    raw = zlib.decompress(payload)
                except zlib.error:
                    raw = payload                           # stored uncompressed
            raw = _exr_unpredict(raw, compression != 0)
            position = 0
            for line in range(lines):
                for name, pixel_type in channels:
                    dtype = np.dtype(_EXR_PIXEL_TYPES[pixel_type])
                    span = width * dtype.itemsize
                    segment = raw[position:position + span]
                    position += span
                    images[name][index * block_lines + line] = \
                        np.frombuffer(segment, dtype=dtype).astype(np.float32)
    return {"size": (width, height), "channels": images,
            "compression": _EXR_COMPRESSION[compression]}


def _srgb_encode(values):
    """sRGB transfer function, the display side of Blender's ``Standard``."""
    import numpy as np

    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92,
                    1.055 * np.power(values, 1.0 / 2.4) - 0.055)


def _display_rgb(linear, exposure: float, transform: str = "Standard"):
    """Scene-linear RGB -> display RGB after ``exposure`` stops."""
    import numpy as np

    scaled = np.clip(linear * (2.0 ** float(exposure)), 0.0, 1.0)
    if transform == "Raw":
        return scaled
    return _srgb_encode(scaled)


def _luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _downsample(array, factor: int):
    """Block-mean downsampling, the grid :func:`_pano_stats` measures on."""
    import numpy as np

    if factor <= 1:
        return array
    rows = array.shape[0] // factor
    cols = array.shape[1] // factor
    if rows < 1 or cols < 1:
        return array
    return array[:rows * factor, :cols * factor].reshape(
        rows, factor, cols, factor, *array.shape[2:]).mean(axis=(1, 3))


def _measure_scale(height: int, width: int) -> int:
    """The block factor :func:`_pano_stats` resizes a panorama with."""
    return max(1, int(math.ceil(max(width, height) / 1024.0)))


def _measured_curve(x, y, bins: int = 48):
    """Monotone display response measured from a (linear, delivered) pair.

    Used when the display transform is not one this module can reproduce
    (AgX and friends): the delivered panorama *is* the response of the
    pipeline's colour management, so its per-bin median against the linear
    radiance gives the curve without knowing the transform.  Returns
    ``(centres, values)`` for :func:`_apply_curve`.
    """
    import numpy as np

    positive = x > 1e-6
    if not positive.any():
        return None
    low = float(np.percentile(x[positive], 0.5))
    high = float(x.max())
    if high <= low:
        return None
    edges = np.geomspace(low, high, bins + 1)
    centres, medians = [], []
    for index in range(bins):
        mask = (x >= edges[index]) & (x < edges[index + 1])
        if not mask.any():
            continue
        centres.append(0.5 * (edges[index] + edges[index + 1]))
        medians.append(float(np.median(y[mask])))
    if len(centres) < 4:
        return None
    values = np.maximum.accumulate(np.asarray(medians))
    return np.asarray(centres), values


def _apply_curve(x, curve):
    import numpy as np

    centres, values = curve
    return np.interp(x, centres, values, left=0.0, right=float(values[-1]))


class _ExposureFrame:
    """One camera's linear EXR, depth map and delivered panorama, on one grid."""

    def __init__(self, camera: str, raw_path: str, pano_path: Optional[str],
                 depth_path: Optional[str], exterior_depth_m: float) -> None:
        import numpy as np

        self.camera = camera
        self.raw_path = raw_path
        self.pano_path = pano_path
        source = read_exr(raw_path)
        names = source["channels"]
        order = [name for name in ("R", "G", "B") if name in names]
        if len(order) < 3:
            order = [name for name, _ in sorted(names.items()) if name != "A"][:3]
        if len(order) < 3:
            raise RuntimeError(f"{raw_path}: fewer than three colour channels")
        height, width = source["size"][1], source["size"][0]
        step = _measure_scale(height, width)
        self.linear = _downsample(np.stack([names[name] for name in order], axis=-1), step)
        self.weights = _erp_weights(int(self.linear.shape[0]))
        self.exterior = self._exterior_mask(depth_path, exterior_depth_m)
        self.delivered = self._delivered_luminance(pano_path)

    def _exterior_mask(self, depth_path: Optional[str], exterior_depth_m: float):
        import numpy as np

        if not depth_path or not os.path.isfile(depth_path):
            return np.zeros(self.linear.shape[:2], dtype=bool)
        try:
            depth = np.load(depth_path).astype("float32")
        except Exception:
            return np.zeros(self.linear.shape[:2], dtype=bool)
        rows, cols = self.linear.shape[:2]
        if depth.shape != (rows, cols):
            factor = max(1, int(round(depth.shape[0] / float(rows))))
            if depth.shape[0] != rows * factor or depth.shape[1] != cols * factor:
                return np.zeros((rows, cols), dtype=bool)
            depth = _downsample(depth, factor)
        return ~np.isfinite(depth) | (depth <= 0.0) | (depth >= exterior_depth_m)

    def _delivered_luminance(self, pano_path: Optional[str]):
        """Display luminance of the delivered panorama on the frame's grid."""
        import numpy as np
        from PIL import Image

        rows, cols = self.linear.shape[:2]
        if not pano_path or not os.path.isfile(pano_path):
            return None
        try:
            with Image.open(pano_path) as image:
                image = image.convert("RGB")
                pixels = np.asarray(image, dtype=np.float32) / 255.0
        except Exception:
            return None
        if pixels.shape[:2] != (rows, cols):
            factor = max(1, int(round(pixels.shape[0] / float(rows))))
            if pixels.shape[0] != rows * factor or pixels.shape[1] != cols * factor:
                return None
            pixels = _downsample(pixels, factor)
        return _luminance(pixels)

    def simulate(self, exposure: float, transform: str):
        """Display luminance of the linear frame after ``exposure`` stops."""
        return _luminance(_display_rgb(self.linear, exposure, transform))

    def metrics(self, display_luminance) -> dict:
        import numpy as np

        clipped = display_luminance >= CLIP_LEVEL
        interior = clipped & ~self.exterior
        exterior = clipped & self.exterior
        return {
            "clip_fraction": round(float(clipped.mean()), 4),
            "interior_clip_fraction": round(_weighted_share(interior, self.weights), 4),
            "exterior_clip_fraction": round(_weighted_share(exterior, self.weights), 4),
            "interior_blob_fraction": round(_largest_blob_fraction(interior), 4),
            "mean_luminance": round(float(display_luminance.mean()), 4),
        }


def _exposure_settings(artifacts: Artifacts) -> dict:
    """The display transform and exposure the delivered panoramas were made with.

    ``run_build.py --no-denoise`` writes a linear EXR and a separate Blender
    pass denoises it with the output node's own colour management
    (``Standard``, exposure 0) -- the ``style.exposure`` of the spec never
    reaches those files.  A build without that post-pass wrote its panoramas
    straight from the render, so the render's own view settings apply.

    ``review_exposure.json`` of an earlier re-do overrides both, but only while
    the panoramas it wrote are still the panoramas on disk; a fresh render
    invalidates it by hash.
    """
    render = ((artifacts.manifest or {}).get("render") or {})
    if render.get("post_denoise"):
        settings = {"transform": "Standard", "exposure": 0.0, "denoised": True,
                    "source": "post_denoise"}
    else:
        settings = {"transform": str(render.get("view_transform") or "Standard"),
                    "exposure": float(render.get("exposure") or 0.0), "denoised": False,
                    "source": "render"}
    applied = _load_json(os.path.join(artifacts.outdir, "review_exposure.json")) or {}
    if applied.get("changed"):
        for camera, record in sorted((applied.get("cameras") or {}).items()):
            panorama = os.path.join(artifacts.outdir, "build", f"pano_{camera}.png")
            if record.get("sha256_after") and _digest(panorama) == record["sha256_after"]:
                settings.update({"exposure": float(applied.get("exposure_after", settings["exposure"])),
                                 "source": "review_exposure.json"})
                break
    return settings


def _digest(path: str) -> Optional[str]:
    """Short content hash of a file, ``None`` when it is not there."""
    import hashlib

    if not os.path.isfile(path):
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()[:16]


def estimate_exposure(artifacts: Artifacts, index: Optional[Dict[str, dict]] = None,
                      target: str = "pass", max_drop: float = 4.0,
                      step: float = EXPOSURE_STEP) -> dict:
    """The one exposure that clears the clipping gate for the whole viewpoint set.

    For every camera the linear EXR is re-exposed on paper (a single global
    stop shift keeps the set photometrically consistent) and the interior
    clip share is measured with the ERP weights; the smallest drop that clears
    the gate wins.  Bright windows are not counted against the estimate -- a
    view whose clipping sits in the sky is reported, not darkened.
    """
    import numpy as np

    settings = _exposure_settings(artifacts)
    result: dict = {
        "kind": "blender_agent_review_exposure_estimate",
        "created_utc": _utc_now(),
        "scene": artifacts.outdir,
        "target": target,
        "transform": settings["transform"],
        "exposure_before": round(float(settings["exposure"]), 3),
        "exposure_after": None,
        "delta_stops": None,
        "cameras": {},
        "warnings": [],
    }
    raw_paths = artifacts.raw_exrs()
    if not raw_paths:
        result.update({"status": "no_linear_source",
                       "detail": "the build kept no linear EXR (render with --no-denoise)"})
        return result
    curve = None
    if settings["transform"] not in _PURE_PYTHON_TRANSFORMS:
        result["warnings"].append(
            f"view transform '{settings['transform']}' is not reproduced by this module; "
            f"the display response is measured from the delivered panoramas instead")
    frames: Dict[str, _ExposureFrame] = {}
    for camera, raw_path in sorted(raw_paths.items()):
        try:
            frames[camera] = _ExposureFrame(camera, raw_path, artifacts.viewpoint_pano(camera),
                                            artifacts.depth_maps().get(camera),
                                            artifacts.exterior_depth_m)
        except Exception as error:
            result["cameras"][camera] = {"error": f"{type(error).__name__}: {error}"}
    usable = {camera: frame for camera, frame in frames.items() if frame.delivered is not None
              or settings["transform"] in _PURE_PYTHON_TRANSFORMS}
    if not usable:
        result.update({"status": "no_linear_source",
                       "detail": "no camera has both a linear EXR and a delivered panorama"})
        return result
    if settings["transform"] not in _PURE_PYTHON_TRANSFORMS:
        scale = 2.0 ** float(settings["exposure"])
        xs = [_luminance(frame.linear).ravel() * scale for frame in usable.values()
              if frame.delivered is not None]
        ys = [frame.delivered.ravel() for frame in usable.values()
              if frame.delivered is not None]
        curve = _measured_curve(np.concatenate(xs), np.concatenate(ys)) if xs else None
        if curve is None:
            result.update({"status": "unsupported_transform",
                           "detail": f"cannot measure the '{settings['transform']}' response"})
            return result

    def evaluate(exposure: float) -> Dict[str, dict]:
        out = {}
        for camera, frame in usable.items():
            if curve is None:
                display = frame.simulate(exposure, settings["transform"])
            else:
                display = _apply_curve(_luminance(frame.linear) * (2.0 ** exposure), curve)
            out[camera] = frame.metrics(display)
        return out

    baseline = evaluate(float(settings["exposure"]))
    result["cameras"] = {camera: {"baseline": metrics} for camera, metrics in baseline.items()}
    model_error = {}
    for camera, frame in usable.items():
        if frame.delivered is None:
            continue
        measured = frame.metrics(frame.delivered)
        model_error[camera] = {
            "measured_interior_clip_fraction": measured["interior_clip_fraction"],
            "modelled_interior_clip_fraction": baseline[camera]["interior_clip_fraction"]}
        result["cameras"][camera]["delivered"] = measured
    if model_error:
        result["model_error"] = model_error
    if target == "warn":
        clip_limit, blob_limit = CLIP_WARN_FRACTION, CLIP_BLOB_WARN
    else:
        clip_limit = CLIP_FAIL_FRACTION * FIX_MARGIN
        blob_limit = CLIP_BLOB_FAIL * FIX_MARGIN
    dark_floor = DARK_WARN_LUMINANCE * 1.25

    def clears(metrics: Dict[str, dict]) -> bool:
        return all(item["interior_clip_fraction"] < clip_limit
                   and item["interior_blob_fraction"] < blob_limit
                   for item in metrics.values())

    def bright_enough(metrics: Dict[str, dict]) -> bool:
        return all(item["mean_luminance"] >= dark_floor for item in metrics.values())

    if clears(baseline):
        result.update({"status": "nothing_to_do", "exposure_after": result["exposure_before"],
                       "delta_stops": 0.0, "predicted": baseline,
                       "detail": "no camera clips indoors beyond the target"})
        return result
    steps = max(1, int(round(max_drop / step)))
    candidates = [settings["exposure"] - round(index * step, 6) for index in range(steps + 1)]
    chosen = chosen_metrics = None
    fallback = None
    for exposure in candidates:
        metrics = evaluate(exposure)
        if not clears(metrics):
            continue
        if fallback is None:
            fallback = (exposure, metrics)
        if bright_enough(metrics):
            chosen, chosen_metrics = exposure, metrics
            break
    if chosen is None:
        if fallback is not None:
            chosen, chosen_metrics = fallback
            result["warnings"].append(
                "the gate is cleared, but at least one viewpoint drops below the dark floor")
        else:
            chosen = candidates[-1]
            chosen_metrics = evaluate(chosen)
            result["warnings"].append(
                f"no drop within {max_drop:.1f} stops clears the gate; "
                f"the scene needs a light or a material fix, not an exposure one")
    result.update({
        "status": "ok" if clears(chosen_metrics) else "unreachable",
        "exposure_after": round(float(chosen), 3),
        "delta_stops": round(float(chosen - settings["exposure"]), 3),
        "predicted": chosen_metrics,
        "failing_cameras": [camera for camera, item in baseline.items()
                            if item["interior_clip_fraction"] >= clip_limit
                            or item["interior_blob_fraction"] >= blob_limit],
    })
    return result


_BLENDER_TIMEOUT_S = 900.0


def _blender_binary() -> Optional[str]:
    """The Blender the pipeline renders with, when it is installed."""
    candidates = [os.environ.get("PANOWORLD_BLENDER"),
                  os.path.join(REPO_ROOT, "third_party", "blender-4.5.14-linux-x64", "blender")]
    import glob as glob_mod
    import shutil

    candidates += sorted(glob_mod.glob(os.path.join(REPO_ROOT, "third_party", "blender-*", "blender")))
    candidates.append(shutil.which("blender"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _blender_rewrite(jobs_path: str) -> None:
    """Run *inside Blender*: denoise + display transform at a new exposure.

    Same chain as ``run_build.py --denoise-only`` (Image -> OpenImageDenoise ->
    File Output with an overridden colour management), which is what turns the
    linear EXR of a render into the delivered panorama.  No geometry, no
    lighting, no ray tracing: the render is re-used, only its colour
    management is redone.
    """
    import bpy
    import glob as glob_mod
    import json as json_mod

    with open(jobs_path, encoding="utf-8") as handle:
        jobs = json_mod.load(handle)
    for job in jobs:
        image = bpy.data.images.load(job["raw"])
        scene = bpy.context.scene
        scene.render.resolution_x = int(image.size[0])
        scene.render.resolution_y = int(image.size[1])
        scene.render.resolution_percentage = 100
        scene.use_nodes = True
        node_tree = scene.node_tree
        node_tree.nodes.clear()
        source = node_tree.nodes.new("CompositorNodeImage")
        source.image = image
        node = source
        if job.get("denoise"):
            denoise = node_tree.nodes.new("CompositorNodeDenoise")
            denoise.prefilter = job["denoise"].get("prefilter", "ACCURATE")
            denoise.quality = job["denoise"].get("quality", "FAST")
            denoise.use_hdr = True
            node_tree.links.new(node.outputs["Image"], denoise.inputs["Image"])
            node = denoise
        output = node_tree.nodes.new("CompositorNodeOutputFile")
        output.base_path = job["work_dir"]
        output.format.file_format = "PNG"
        output.format.color_mode = "RGB"
        output.format.color_depth = "8"
        if hasattr(output.format, "color_management"):
            output.format.color_management = "OVERRIDE"
            output.format.view_settings.view_transform = str(job.get("view_transform", "Standard"))
            output.format.view_settings.look = "None"
            output.format.view_settings.exposure = float(job.get("exposure", 0.0))
        stem = f"{job['camera']}_exposure_"
        output.file_slots[0].path = stem
        node_tree.links.new(node.outputs["Image"], output.inputs[0])
        bpy.ops.render.render(write_still=False)
        written = sorted(glob_mod.glob(os.path.join(job["work_dir"], f"{stem}*.png")))
        if not written:
            raise RuntimeError(f"{job['raw']}: the colour-management pass wrote no image")
        os.replace(written[-1], job["out"])
        for stale in written[:-1]:
            os.remove(stale)
        bpy.data.images.remove(image)


def _run_blender_rewrite(jobs: List[dict], blender: str) -> None:
    import tempfile

    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    try:
        json.dump(jobs, handle)
        handle.close()
        expression = (f"import sys; sys.path.insert(0, {REPO_ROOT!r}); "
                      f"from blender_agent import review; review._blender_rewrite({handle.name!r})")
        result = subprocess.run([blender, "--background", "--factory-startup",
                                 "--python-expr", expression],
                                capture_output=True, text=True, timeout=_BLENDER_TIMEOUT_S, cwd=REPO_ROOT)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-600:]
            raise RuntimeError(f"blender exited with {result.returncode}: {tail}")
        missing = [job["out"] for job in jobs if not os.path.isfile(job["out"])]
        if missing:
            raise RuntimeError(f"blender left no panorama for {missing}")
    finally:
        os.unlink(handle.name)


def _python_rewrite(frame: "_ExposureFrame", exposure: float, transform: str, out_path: str) -> dict:
    """Re-expose the linear EXR without Blender: the same display transform,
    no denoise (the delivered panorama came from a denoised render, so this
    fallback trades a little noise for not needing a renderer)."""
    import numpy as np
    from PIL import Image

    source = read_exr(frame.raw_path)
    names = source["channels"]
    channels = [names[name] for name in ("R", "G", "B") if name in names]
    if len(channels) < 3:
        channels = [value for name, value in sorted(names.items()) if name != "A"][:3]
    display = _display_rgb(np.stack(channels, axis=-1), exposure, transform)
    Image.fromarray((display * 255.0 + 0.5).astype("uint8")).save(out_path)
    return {"pixels": "linear-exr", "denoised": False,
            "view_transform": transform, "exposure": round(float(exposure), 4)}


def fix_exposure(outdir: str, spec_path: Optional[str] = None, manifest_path: Optional[str] = None,
                 index: Optional[Dict[str, dict]] = None, target: str = "pass",
                 max_drop: float = 4.0, use_blender: bool = True, dry_run: bool = False,
                 review_after: bool = True, review_cmd: Optional[str] = None,
                 allow_external_error: bool = False, strict: bool = False,
                 report_path: Optional[str] = None) -> dict:
    """Redo the colour management of an existing render at a better exposure.

    The benchmark's exposure repair used to subtract a fixed 0.3 stops from
    ``style.exposure`` and re-render -- which neither measures what the
    panorama does nor, in the ``--no-denoise`` flow, reaches the delivered
    image at all.  This instead estimates the exposure from the *linear* EXRs
    (interior clipping only, ERP-weighted, one value for the whole viewpoint
    set) and redoes the colour management -- denoise and display transform --
    on the saved render: no modelling, no settling, no depth rays, minutes
    instead of a full rebuild.  ``spec.json`` is left alone; the images are
    re-derived from the same build, which is why ``review_exposure.json``
    records what changed and ``build/pano_pre_exposure/`` keeps the old files.

    Returns the estimate, the per-camera rewrite report and, unless
    ``review_after`` is off, the re-review of the images that were written.
    """
    import shutil

    started = time.time()
    artifacts = Artifacts(outdir, spec_path, manifest_path)
    estimate = estimate_exposure(artifacts, index=index, target=target, max_drop=max_drop)
    result: dict = {
        "kind": "blender_agent_review_exposure",
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "scene": artifacts.outdir,
        "target": target,
        "dry_run": bool(dry_run),
        "estimate": estimate,
        "cameras": {},
        "changed": False,
    }
    if estimate.get("status") not in ("ok", "unreachable"):
        result["status"] = estimate.get("status") or "error"
        result["detail"] = estimate.get("detail") or estimate.get("warnings")
        return result
    settings = _exposure_settings(artifacts)
    exposure = estimate["exposure_after"]
    blender = _blender_binary() if use_blender else None
    pure_python = settings["transform"] in _PURE_PYTHON_TRANSFORMS
    if blender is None and not pure_python:
        result.update({"status": "unsupported_transform",
                       "detail": f"view transform '{settings['transform']}' needs Blender "
                                 f"for the colour re-do and none was found"})
        return result
    raw_paths = artifacts.raw_exrs()
    build_dir = os.path.join(artifacts.outdir, "build")
    backup_dir = os.path.join(build_dir, "pano_pre_exposure")
    denoise = None
    if settings["denoised"] and blender is not None:
        recorded = (artifacts.manifest.get("render") or {}).get("post_denoise") or {}
        denoise = {"prefilter": recorded.get("prefilter", "ACCURATE"),
                   "quality": recorded.get("quality", "FAST")}
    jobs: List[dict] = []
    frames: Dict[str, "_ExposureFrame"] = {}
    for camera, raw_path in sorted(raw_paths.items()):
        pano = os.path.join(build_dir, f"pano_{camera}.png")
        if not os.path.isfile(pano):
            result["cameras"][camera] = {"status": "skipped", "why": "no build/pano_<cam>.png"}
            continue
        view = artifacts.viewpoint_pano(camera)
        if view is None:
            result["cameras"][camera] = {"status": "skipped", "why": "no exported panoImage"}
            continue
        if blender is not None:
            jobs.append({"camera": camera, "raw": raw_path, "out": pano,
                         "work_dir": build_dir, "view_transform": settings["transform"],
                         "exposure": float(exposure), "denoise": denoise})
        else:
            try:
                frames[camera] = _ExposureFrame(camera, raw_path, view,
                                                artifacts.depth_maps().get(camera),
                                                artifacts.exterior_depth_m)
            except Exception as error:
                result["cameras"][camera] = {"status": "error",
                                             "why": f"{type(error).__name__}: {error}"}
    if not jobs and not frames:
        result.update({"status": "error", "detail": "no camera could be re-exposed"})
        return result

    before = {camera: _digest(os.path.join(build_dir, f"pano_{camera}.png"))
              for camera in list(raw_paths) if os.path.isfile(os.path.join(build_dir,
                                                                          f"pano_{camera}.png"))}
    if not dry_run:
        os.makedirs(backup_dir, exist_ok=True)
        for camera in before:
            for source, name in ((os.path.join(build_dir, f"pano_{camera}.png"),
                                  f"pano_{camera}.png"),
                                 (artifacts.viewpoint_pano(camera), f"viewpoint_{camera}.png")):
                if source and os.path.isfile(source):
                    target = os.path.join(backup_dir, name)
                    if not os.path.isfile(target):
                        shutil.copy2(source, target)
    if dry_run:
        result.update({"status": "dry_run", "detail": "the estimate is ready, nothing was written",
                       "exposure_before": estimate["exposure_before"],
                       "exposure_after": exposure, "seconds": round(time.time() - started, 3)})
        return result
    try:
        if jobs:
            _run_blender_rewrite(jobs, blender)
        for camera, frame in frames.items():
            _python_rewrite(frame, exposure, settings["transform"],
                            os.path.join(build_dir, f"pano_{camera}.png"))
    except Exception as error:
        result.update({"status": "error", "detail": f"{type(error).__name__}: {error}",
                       "backup": backup_dir})
        return result
    for camera in sorted(set(list(before) + [job["camera"] for job in jobs] + list(frames))):
        pano = os.path.join(build_dir, f"pano_{camera}.png")
        if not os.path.isfile(pano):
            continue
        view = artifacts.viewpoint_pano(camera)
        if view and not dry_run:
            shutil.copyfile(pano, view)
        record = {"status": "written", "panorama": os.path.relpath(pano, artifacts.outdir),
                  "exposure": round(float(exposure), 4),
                  "sha256_before": before.get(camera), "sha256_after": _digest(pano),
                  "denoised": bool(denoise and blender is not None),
                  "view_transform": settings["transform"]}
        if camera in estimate.get("predicted", {}):
            record["predicted"] = estimate["predicted"][camera]
        result["cameras"][camera] = record
    result.update({"changed": True, "status": "applied",
                   "exposure_before": estimate["exposure_before"], "exposure_after": exposure,
                   "delta_stops": estimate["delta_stops"],
                   "style_exposure_recommended": round(float(exposure), 3),
                   "backup": backup_dir})
    exposure_path = os.path.join(artifacts.outdir, "review_exposure.json")
    if review_after:
        report = review(artifacts.outdir, spec_path=artifacts.spec_path,
                        manifest_path=artifacts.manifest_path, index=index,
                        review_cmd=review_cmd, strict=strict,
                        allow_external_error=allow_external_error)
        result["review"] = {"status": report.get("status"), "pass": report.get("pass"),
                            "score": report.get("score"), "missing": report.get("missing"),
                            "issues": report.get("stats"),
                            "failed_checks": [check["name"] for check in report.get("checks", [])
                                              if not check["pass"]]}
        target_report = report_path or os.path.join(artifacts.outdir, DEFAULT_REPORT_NAME)
        try:
            with open(target_report, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        except OSError:
            pass
    result["seconds"] = round(time.time() - started, 3)
    try:
        with open(exposure_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    except OSError:
        pass
    return result


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
            estimate = None
            try:                        # the linear EXRs say how much is too much
                estimate = estimate_exposure(artifacts, index=index)
            except Exception:
                estimate = None
            status = (estimate or {}).get("status")
            if raw.get("style") and (estimate is None or status == "no_linear_source"):
                after = round(max(0.0, exposure - 0.3), 3)
                raw["style"] = dict(style, exposure=after)
                spec = _load_from_dict(raw)
                actions.append({"kind": "reduce_exposure", "exposure_before": exposure,
                                "exposure_after": after,
                                "measured": "fixed trim (no linear EXR to measure)",
                                "why": "a panorama was blown out"})
            elif raw.get("style") and status in ("ok", "unreachable"):
                raw["style"] = dict(style, exposure=estimate["exposure_after"])
                spec = _load_from_dict(raw)
                actions.append({"kind": "reduce_exposure", "exposure_before": exposure,
                                "exposure_after": estimate["exposure_after"],
                                "delta_stops": estimate["delta_stops"],
                                "measured": "linear EXR statistics",
                                "failing_cameras": estimate.get("failing_cameras") or [],
                                "why": "a panorama clips indoors"})
            else:
                skipped.append({"kind": "reduce_exposure", "why": status or "unavailable",
                                "detail": (estimate or {}).get("detail")})

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
    parser.add_argument("--no-build", action="store_true",
                        help="the run asked for no render (layout-only): a missing build is a "
                             "requested skip, not a missing required input "
                             "(also PANOWORLD_SKIP_RENDER=1)")
    parser.add_argument("--no-images", action="store_true", help="skip the panorama pixel checks")
    parser.add_argument("--no-depth", action="store_true", help="skip the depth-map reads")
    parser.add_argument("--strict", action="store_true", help="warnings fail the review too")
    parser.add_argument("--review-cmd", default=os.environ.get("REVIEW_CMD"),
                        help="external reviewer (default: $REVIEW_CMD)")
    parser.add_argument("--timeout", type=float, default=180.0, help="external reviewer timeout (s)")
    parser.add_argument("--allow-external-error", action="store_true",
                        default=os.environ.get("PANOWORLD_REVIEW_ALLOW_EXTERNAL_ERROR", "") == "1",
                        help="an external reviewer that cannot be reached degrades to a warning "
                             "(default: it is a retryable error and the review does not pass)")
    parser.add_argument("--apply-repair", action="store_true",
                        help="patch spec.json from review_report.json (exit 0 when it changed)")
    parser.add_argument("--fix-exposure", action="store_true",
                        help="redo the colour management of the existing render from the saved "
                             "linear EXRs at a measured exposure (no re-render)")
    parser.add_argument("--exposure-target", choices=["pass", "warn"], default="pass",
                        help="with --fix-exposure: clear the gate (default) or the warning level")
    parser.add_argument("--exposure-max-drop", type=float, default=4.0,
                        help="with --fix-exposure: stop searching after this many stops")
    parser.add_argument("--no-blender", action="store_true",
                        help="with --fix-exposure: never call Blender; only a Standard view "
                             "transform can be redone without it")
    parser.add_argument("--no-review", dest="review_after", action="store_false",
                        help="with --fix-exposure: do not re-review the images written")
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

    if args.fix_exposure:
        try:
            result = fix_exposure(outdir, spec_path=args.spec, manifest_path=args.manifest,
                                  target=args.exposure_target, max_drop=args.exposure_max_drop,
                                  use_blender=not args.no_blender, dry_run=args.dry_run,
                                  review_after=args.review_after, review_cmd=args.review_cmd,
                                  allow_external_error=args.allow_external_error,
                                  strict=args.strict, report_path=args.json_out)
        except Exception as error:                       # a failed re-do must stay legible
            import traceback

            result = {"kind": "blender_agent_review_exposure", "changed": False,
                      "status": "error", "error": f"{type(error).__name__}: {error}",
                      "traceback": traceback.format_exc()[-2000:]}
        if not args.quiet:
            estimate = result.get("estimate") or {}
            print(f"review exposure: {result['status']} "
                  f"(before {result.get('exposure_before', estimate.get('exposure_before'))}, "
                  f"after {result.get('exposure_after', estimate.get('exposure_after'))})")
            for camera, record in sorted((result.get("cameras") or {}).items()):
                print(f"  {camera}: {json.dumps(record)}")
            for warning in (estimate.get("warnings") or []):
                print(f"  warning: {warning}")
            if result.get("review"):
                print(f"  re-review: {result['review']['status']} "
                      f"({result['review']['issues']})")
            if result.get("detail"):
                print(f"  detail: {result['detail']}")
            if result.get("error"):
                print(f"  error: {result['error']}")
        return 0 if result.get("changed") else (2 if result.get("status") == "error" else 1)

    try:
        report = review(outdir, spec_path=args.spec, manifest_path=args.manifest,
                        images=not args.no_images, depth=not args.no_depth,
                        review_cmd=args.review_cmd, strict=args.strict,
                        spec_only=args.spec_only, timeout=args.timeout,
                        allow_external_error=args.allow_external_error,
                        no_build=args.no_build or os.environ.get("PANOWORLD_SKIP_RENDER", "") == "1")
    except Exception as error:                           # a review crash must be legible
        import traceback

        report = {"kind": "blender_agent_review", "schema_version": SCHEMA_VERSION,
                  "created_utc": _utc_now(), "scene": outdir, "pass": False, "status": "error",
                  "score": 0.0, "missing": ["internal_error"],
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
