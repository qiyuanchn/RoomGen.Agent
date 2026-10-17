"""Structural diversity of generated houses -- and the 100-seed benchmark.

`layout_gen.py` can produce a thousand specs; the question this tool answers is
whether they are a thousand *different* houses or the same corridor with other
furniture.  It reads specs (a batch root, explicit paths, or seeds it generates)
and measures, per scene:

| metric | of which |
|---|---|
| `rooms`, `room_types`, `floors` | room count, type histogram, storey count |
| `structure` | the **attributed room graph**: one node per room *instance* (function type + storey), one edge per connection (`door` / `open_passage` / `stair` / `ramp`), its cycle rank, and a canonical fingerprint of that graph |
| `shape` | bounding box, **union footprint** area, per-storey areas, total floor area, aspect ratio |
| `functions` | which room types the plan covers |
| `furniture` | item count, distinct assets, parts/table densities, per-room-type asset sets, combo hash |
| `cameras` | per-room counts, pairwise spacing in **3D**, spacing within one storey, navigation distance through the room graph, unreachable pairs, rooms without a viewpoint |
| `lighting` | sun elevation/azimuth, sky strength, exposure, windows |

and then aggregates them: histograms, per-family counts, and **topology
repeats** -- scenes whose canonical structure fingerprint is identical.  Two
houses are the same topology when the same functions sit on the same storeys
with the same connections and cycle rank, no matter how the rooms are named, in
which order the lists were written, or which furniture stands in them: a chain,
a star and a ring of four rooms are three topologies, and a stair that joins two
storeys is part of the structure instead of a missing edge.  The fingerprint's
hash only groups *candidates*; membership in a group is decided by comparing the
fingerprints themselves, so different graphs that happen to share a hash stay
apart (`hash_collisions` counts how often that happened).  `plan_hash` is the
stricter neighbour (structure + coarse shape bins) for "near-identical plan"
detection.

Connections come from the spec contract: `connectors` **and** the legacy
`openings`, merged the same way the builder merges them (a spec that does not
parse falls back to a tolerant raw reading).  Areas are union areas of the room
outlines, sampled per compressed x-strip -- exact for the rectilinear polygons
the generator writes, a midpoint estimate for oblique ones -- so an L-shaped or
stacked plan no longer reports its bounding rectangle as its floor space.

Command line::

    # explicit specs / batch roots / layout_report.json files
    python tools/diversity_report.py specs outputs/stream/*/spec.json --out /tmp/diversity.json

    # generate in process and measure (no files written unless --out is given)
    python tools/diversity_report.py seeds --seeds 0-99 --out /tmp/diversity100.json

    # the batch benchmark: 100 seeds of layout+diversity, a stratified sample
    # rendered and reviewed, collated into one report
    python tools/diversity_report.py benchmark --seeds 0-99 --sample 10 \
        --root outputs/benchmark100

    # the *delivered* products of a batch: requested vs built vs surviving
    # families, fallbacks, floors, layout shape, door graph, asset reuse and
    # camera view coverage -- reading batch_summary.json plus each scene's
    # spec/layout_report/run_summary
    python tools/diversity_report.py success \
        --root outputs/benchmark100/layout --root outputs/benchmark100/sample \
        --out outputs/benchmark100/diversity_success.json

The `success` mode is the one that answers "did we deliver a diverse dataset?"
rather than "does the generator cover a lot?": it grades the request, the built
specs and the final ``bundle_ok`` products separately, so a hard family that
falls back (``stats.auto_pick`` -> ``stats.auto_fallback``) or a product the
pipeline filtered (a failed review, an interrupted render) shows up as a mix
that moved instead of hiding inside a request count.  Legacy artifacts that do
not record the drawn family are re-drawn with the current generator and marked
``replayed``, never counted as "asked for what it built".

The `benchmark` mode drives the real scripts (`scripts/batch_generate.sh`), so
its timings are the pipeline's own and its failure distribution comes from the
same grading the batch prints: `layout_ok` / `render_ok` / `validation_ok` /
`review_ok` / `bundle_ok`.  Rendering is the expensive part, so the sample is
drawn deliberately: at least one scene per topology family, then seeds that add
a *new* structure fingerprint, filled up in seed order.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import random

import statistics
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_BATCH = os.path.join(REPO_ROOT, "scripts", "batch_generate.sh")
DEFAULT_HOSTPY = os.environ.get("PANOWORLD_HOSTPY", "/opt/conda/envs/diffaero/bin/python")

TRANSIT_TYPES = ("corridor", "balcony", "entry")

#: connection kinds the spec contract carries (see blender_agent/spec.py)
CONNECTION_KINDS = ("door", "open_passage", "stair", "ramp")
CROSS_FLOOR_KINDS = ("stair", "ramp")

#: world z of a camera that only carries an XY position (legacy specs)
DEFAULT_CAMERA_Z_M = 1.5

#: individualisation/refinement visits budget for one structure fingerprint
CANONICAL_BUDGET = 4096

#: stage grades a batch row and a run summary share, weakest first; the success
#: report walks them as a funnel instead of collapsing everything into "passed"
SUCCESS_LEVELS = ("layout_ok", "render_ok", "validation_ok", "bundle_ok")

#: seed salt of layout_gen's per-attempt rng; the ``--shape auto`` draw lives on
#: its own stream ``seed * ATTEMPT_SALT + AUTO_PICK_SALT`` (layout_gen.generate)
ATTEMPT_SALT = 1000003

#: bins of the shape histograms: what a mean aspect or area hides
ASPECT_BINS = (1.0, 1.5, 2.0, 3.0)
AREA_BINS = (0.0, 40.0, 80.0, 120.0)


# --------------------------------------------------------------------------- #
# spec reading / small helpers
# --------------------------------------------------------------------------- #
def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _round(value: float, step: float) -> float:
    return round(value / step) * step


def _as_list(value) -> list:
    """A JSON list, or nothing: one malformed block must not sink the report."""
    return value if isinstance(value, list) else []


def load_specs(paths: Sequence[str]) -> List[dict]:
    """Expand paths/dirs/globs into ``[{"path", "spec", "layout_report"}]``."""
    out: List[dict] = []
    seen = set()
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "spec.json"))))
            files.extend(sorted(glob.glob(os.path.join(path, "scene_*", "spec.json"))))
        else:
            files.extend(sorted(glob.glob(path)) or [path])
    for path in files:
        path = os.path.abspath(path)
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            with open(path, encoding="utf-8") as handle:
                spec = json.load(handle)
        except Exception:
            continue
        if not isinstance(spec, dict):
            continue
        report_path = os.path.join(os.path.dirname(path), "layout_report.json")
        report = None
        if os.path.isfile(report_path):
            try:
                with open(report_path, encoding="utf-8") as handle:
                    report = json.load(handle)
            except Exception:
                report = None
        out.append({"path": path, "spec": spec, "layout_report": report})
    return out


def parse_seeds(text: str) -> List[int]:
    """``"0-99"`` / ``"0,7,13"`` / ``"5-9,20"`` -> sorted ints."""
    seeds: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            low, _, high = chunk.partition("-")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(chunk))
    return sorted(dict.fromkeys(seeds))


def _spec_module():
    """``blender_agent.spec`` when importable (it is pure python, no bpy)."""
    try:
        from blender_agent import spec as spec_mod  # noqa: PLC0415 - optional import
    except Exception:
        return None
    return spec_mod


# --------------------------------------------------------------------------- #
# the room graph of one scene
# --------------------------------------------------------------------------- #
def scene_graph(spec: dict) -> dict:
    """Rooms, storeys and *every* connection of one spec dict.

    The frozen contract in ``blender_agent.spec`` is preferred, so ``openings``
    and ``connectors`` end up in one graph exactly the way the builder merges
    them; a spec that does not parse falls back to a tolerant raw reading.

    Returns ``rooms`` (``id``/``type``/``floor`` index/``polygon``), ``floor_ids``
    and ``floor_elevations`` in spec order, ``connections`` as
    ``(from_room, to_room, kind)``, ``cameras`` with a 3D position, and which
    reader produced it.
    """
    module = _spec_module()
    if module is not None:
        try:
            parsed = module.parse_spec(spec)
            floor_ids = [str(floor.id) for floor in parsed.floors]
            floor_index = {floor_id: index for index, floor_id in enumerate(floor_ids)}
            rooms = [{"id": str(room.id), "type": str(room.type),
                      "floor": floor_index.get(str(room.floor_id), 0 if not floor_ids else -1),
                      "polygon": [[float(point[0]), float(point[1])] for point in room.polygon]}
                     for room in parsed.rooms]
            cameras = [{"name": str(camera.name), "room": str(camera.room),
                        "pos": [float(value) for value in camera.pos]}
                       for camera in parsed.cameras]
            return {
                "rooms": rooms,
                "floor_ids": floor_ids,
                "floor_elevations": [float(floor.elevation) for floor in parsed.floors],
                "connections": [(str(conn.from_room), str(conn.to_room), str(conn.kind))
                                for conn in parsed.connections()],
                "cameras": cameras,
                "source": "spec",
            }
        except Exception:
            pass
    return _raw_scene_graph(spec)


def _raw_scene_graph(spec: dict) -> dict:
    raw_floors = _as_list(spec.get("floors"))
    floor_ids = [str(floor.get("id")) for floor in raw_floors if isinstance(floor, dict)]
    floor_elevations = [float(floor.get("elevation", 0.0)) for floor in raw_floors
                        if isinstance(floor, dict)]
    floor_index = {floor_id: index for index, floor_id in enumerate(floor_ids)}
    rooms = []
    for room in _as_list(spec.get("rooms")):
        if not isinstance(room, dict):
            continue
        floor_id = room.get("floor_id")
        if floor_id is None:
            index = 0
        else:
            index = floor_index.setdefault(str(floor_id), len(floor_index))
        rooms.append({"id": str(room.get("id")), "type": str(room.get("type", "unknown")),
                      "floor": index,
                      "polygon": [[float(point[0]), float(point[1])]
                                  for point in _as_list(room.get("polygon"))
                                  if len(point) >= 2]})
    known = {room["id"] for room in rooms}
    connections: List[Tuple[str, str, str]] = []
    raw_connectors = spec.get("connectors")
    if raw_connectors is None:
        raw_connectors = spec.get("connector")
    for connector in _as_list(raw_connectors):
        if not isinstance(connector, dict):
            continue
        first, second = str(connector.get("from_room")), str(connector.get("to_room"))
        if first in known and second in known:
            connections.append((first, second, str(connector.get("kind", "door"))))
    for opening in _as_list(spec.get("openings")):
        if not isinstance(opening, dict):
            continue
        between = [str(value) for value in _as_list(opening.get("between"))]
        if len(between) == 2 and between[0] in known and between[1] in known:
            connections.append((between[0], between[1], str(opening.get("kind", "door"))))
    cameras = []
    for camera in _as_list(spec.get("cameras")):
        if not isinstance(camera, dict):
            continue
        cameras.append({"name": str(camera.get("name")), "room": str(camera.get("room")),
                        "pos": [float(value) for value in _as_list(camera.get("pos"))]})
    return {
        "rooms": rooms,
        "floor_ids": floor_ids,
        "floor_elevations": floor_elevations,
        "connections": connections,
        "cameras": cameras,
        "source": "raw",
    }


def floor_label(graph: dict, index: int) -> str:
    """Readable storey name: the declared floor id, else ``floor<index>``."""
    labels = graph["floor_ids"]
    if 0 <= index < len(labels):
        return str(labels[index])
    return "unassigned" if index < 0 else f"floor{index}"


def floor_elevation(graph: dict, index: int) -> float:
    elevations = graph["floor_elevations"]
    if 0 <= index < len(elevations):
        return elevations[index]
    return 0.0


# --------------------------------------------------------------------------- #
# areas: union of room outlines, not the bounding rectangle
# --------------------------------------------------------------------------- #
def _polygon_spans(polygon: Sequence[Sequence[float]], x: float) -> List[Tuple[float, float]]:
    """The y intervals a vertical line at ``x`` cuts out of one polygon."""
    crossings: List[float] = []
    count = len(polygon)
    for index in range(count):
        x1, y1 = float(polygon[index][0]), float(polygon[index][1])
        x2, y2 = float(polygon[(index + 1) % count][0]), float(polygon[(index + 1) % count][1])
        if (x1 <= x < x2) or (x2 <= x < x1):
            crossings.append(y1 + (y2 - y1) * (x - x1) / (x2 - x1))
    crossings.sort()
    return [(crossings[i], crossings[i + 1]) for i in range(0, len(crossings) - 1, 2)]


def _union_span_length(spans: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    end: Optional[float] = None
    for start, stop in sorted(spans):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def polygon_union_area(polygons: Sequence[Sequence[Sequence[float]]]) -> float:
    """Area of the union of the outlines.

    The x axis is split at every vertex coordinate and each strip contributes
    ``width * union of the y intervals at the strip centre``.  That is exact for
    rectilinear outlines (what the families and the duplex generator write) and
    a midpoint estimate for oblique ones; overlapping rooms are counted once.
    """
    shapes = [[[float(point[0]), float(point[1])] for point in polygon]
              for polygon in polygons if len(polygon) >= 3]
    if not shapes:
        return 0.0
    xs = sorted({point[0] for shape in shapes for point in shape})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right - left <= 1e-9:
            continue
        middle = 0.5 * (left + right)
        spans: List[Tuple[float, float]] = []
        for shape in shapes:
            spans.extend(_polygon_spans(shape, middle))
        area += (right - left) * _union_span_length(spans)
    return area


def _centroid(polygon: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """Area centroid of a simple polygon (falls back to the vertex mean)."""
    count = len(polygon)
    if count == 0:
        return 0.0, 0.0
    twice_area = 0.0
    cx = cy = 0.0
    for index in range(count):
        x1, y1 = float(polygon[index][0]), float(polygon[index][1])
        x2, y2 = float(polygon[(index + 1) % count][0]), float(polygon[(index + 1) % count][1])
        cross = x1 * y2 - x2 * y1
        twice_area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(twice_area) <= 1e-9:
        return (sum(float(point[0]) for point in polygon) / count,
                sum(float(point[1]) for point in polygon) / count)
    return cx / (3.0 * twice_area), cy / (3.0 * twice_area)


# --------------------------------------------------------------------------- #
# canonical fingerprint of the attributed room graph
# --------------------------------------------------------------------------- #
def _refine(node_count: int, adjacency: Sequence[Sequence[Tuple[int, str]]],
            colors: Sequence[int]) -> List[int]:
    """One canonical colour-refinement pass, repeated until the partition holds."""
    while True:
        signatures = []
        for node in range(node_count):
            neighbours = sorted((kind, colors[other]) for other, kind in adjacency[node])
            signatures.append((colors[node], tuple(neighbours)))
        unique = sorted(set(signatures))
        index = {signature: position for position, signature in enumerate(unique)}
        refined = [index[signature] for signature in signatures]
        if len(unique) == len(set(colors)):
            return refined
        colors = refined


def _certificate(node_count: int, adjacency: Sequence[Sequence[Tuple[int, str]]],
                 node_attrs: Sequence[tuple], colors: Sequence[int]) -> str:
    """Full graph serialisation in canonical (colour) order -- all colours distinct."""
    order = sorted(range(node_count), key=lambda node: (colors[node], node))
    rank = {node: position for position, node in enumerate(order)}
    payload = []
    for node in order:
        payload.append([list(node_attrs[node]),
                        sorted([kind, rank[other]] for other, kind in adjacency[node])])
    return json.dumps(payload, separators=(",", ":"))


def _quotient_key(node_count: int, adjacency: Sequence[Sequence[Tuple[int, str]]],
                  node_attrs: Sequence[tuple], colors: Sequence[int]) -> str:
    """Invariant but possibly incomplete key: the coloured quotient graph."""
    payload = sorted([list(node_attrs[node]),
                      sorted([kind, colors[other]] for other, kind in adjacency[node])]
                     for node in range(node_count))
    return json.dumps(payload, separators=(",", ":"))


def canonical_graph_key(node_attrs: Sequence[tuple],
                        edges: Sequence[Tuple[int, int, str]],
                        floor_count: int) -> Tuple[str, bool]:
    """Canonical fingerprint of the attributed room graph, and whether it is exact.

    ``node_attrs`` are per room instance (function, storey index), ``edges`` are
    ``(node, node, kind)`` with ``kind`` a connection type.  Colour refinement
    plus individualisation search gives a canonical form: it does not depend on
    room names, node or edge order, and a chain, a star and a ring of the same
    rooms produce different keys.  ``exact`` is false when the search budget ran
    out and the key fell back to the refined quotient (still order invariant,
    but two non-isomorphic graphs may then share it).
    """
    node_count = len(node_attrs)
    adjacency: List[List[Tuple[int, str]]] = [[] for _ in range(node_count)]
    for first, second, kind in edges:
        if first == second:
            continue
        adjacency[first].append((second, str(kind)))
        adjacency[second].append((first, str(kind)))
    attrs = [tuple(attr) for attr in node_attrs]
    distinct = sorted(set(attrs))
    initial = [distinct.index(attr) for attr in attrs]
    state = {"visits": 0, "exact": True}

    def explore(colors: List[int]) -> str:
        state["visits"] += 1
        if state["visits"] > CANONICAL_BUDGET:
            state["exact"] = False
            return _quotient_key(node_count, adjacency, attrs, colors)
        colors = _refine(node_count, adjacency, colors)
        classes: Dict[int, List[int]] = {}
        for node, color in enumerate(colors):
            classes.setdefault(color, []).append(node)
        ties = sorted(color for color, members in classes.items() if len(members) > 1)
        if not ties:
            return _certificate(node_count, adjacency, attrs, colors)
        fresh = max(colors) + 1
        best: Optional[str] = None
        for node in classes[ties[0]]:
            child = list(colors)
            child[node] = fresh
            key = explore(child)
            if best is None or key < best:
                best = key
        return best

    certificate = explore(initial)
    marker = "exact" if state["exact"] else "quotient"
    return f"floors={int(floor_count)}|{marker}|{certificate}", bool(state["exact"])


def _cycle_rank(node_count: int, edges: Sequence[Tuple[int, int, str]]) -> int:
    """``E - V + C`` of the room graph: independent loops of the plan."""
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first, second, _ in edges:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_a] = root_b
    if node_count == 0:
        return 0
    components = len({find(node) for node in range(node_count)})
    return max(0, len(edges) - node_count + components)


# --------------------------------------------------------------------------- #
# per-scene metrics
# --------------------------------------------------------------------------- #
def _camera_metrics(graph: dict, rooms: Sequence[dict], room_by_id: Dict[str, dict],
                    edges: Sequence[Tuple[str, str, str]]) -> dict:
    """Per-room counts plus 3D, same-storey and navigation camera distances."""
    cameras = graph["cameras"]
    per_room: Dict[str, int] = {}
    positions: List[Tuple[float, float, float]] = []
    camera_rooms: List[str] = []
    floors: List[int] = []
    for camera in cameras:
        room_id = camera["room"]
        per_room[room_id] = per_room.get(room_id, 0) + 1
        room = room_by_id.get(room_id)
        position = list(camera["pos"])
        if len(position) >= 3:
            z = float(position[2])
        else:
            z = floor_elevation(graph, room["floor"] if room else 0) + DEFAULT_CAMERA_Z_M
        x = float(position[0]) if position else 0.0
        y = float(position[1]) if len(position) > 1 else 0.0
        positions.append((x, y, z))
        camera_rooms.append(room_id)
        floors.append(room["floor"] if room else 0)

    gaps_3d: List[float] = []
    gaps_same_floor: List[float] = []
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            gap = math.dist(positions[first], positions[second])
            gaps_3d.append(gap)
            if floors[first] == floors[second]:
                gaps_same_floor.append(gap)

    navigation = _navigation_distances(graph, rooms, room_by_id, edges, positions, camera_rooms)

    return {
        "count": len(cameras),
        "per_room": per_room,
        "floor_counts": {floor_label(graph, floor): sum(1 for value in floors if value == floor)
                         for floor in sorted(set(floors))},
        "rooms_without_camera": sorted(room["id"] for room in rooms
                                       if room["id"] not in per_room
                                       and room.get("type") not in TRANSIT_TYPES),
        "min_gap_3d_m": round(min(gaps_3d), 3) if gaps_3d else None,
        "mean_gap_3d_m": round(statistics.fmean(gaps_3d), 3) if gaps_3d else None,
        "min_gap_same_floor_m": round(min(gaps_same_floor), 3) if gaps_same_floor else None,
        "mean_gap_same_floor_m": round(statistics.fmean(gaps_same_floor), 3)
        if gaps_same_floor else None,
        "min_gap_nav_m": round(min(navigation["gaps"]), 3) if navigation["gaps"] else None,
        "mean_gap_nav_m": round(statistics.fmean(navigation["gaps"]), 3)
        if navigation["gaps"] else None,
        "unreachable_pairs": navigation["unreachable"],
    }


def _navigation_distances(graph: dict, rooms: Sequence[dict], room_by_id: Dict[str, dict],
                          edges: Sequence[Tuple[str, str, str]],
                          positions: Sequence[Tuple[float, float, float]],
                          camera_rooms: Sequence[str]) -> dict:
    """Camera-to-camera distance *through the plan*, via the room graph.

    A camera pair in one room is the direct 3D distance (no wall between them);
    otherwise the walk is camera -> its room centre -> shortest connection path
    -> room centre -> camera.  That is a lower bound on the real walking
    distance (it ignores furniture and where the door sits on the wall) but it
    never reports two storeys as zero metres apart, which an XY-only spacing did.
    """
    centroids: Dict[str, Tuple[float, float, float]] = {}
    for room in rooms:
        x, y = _centroid(room["polygon"])
        centroids[room["id"]] = (x, y, floor_elevation(graph, room["floor"]))
    weights: Dict[str, Dict[str, float]] = {}
    for first, second, _kind in edges:
        if first == second or first not in centroids or second not in centroids:
            continue
        span = math.dist(centroids[first], centroids[second])
        for node, other in ((first, second), (second, first)):
            best = weights.setdefault(node, {}).get(other)
            if best is None or span < best:
                weights[node][other] = span

    def shortest(source: str) -> Dict[str, float]:
        distances: Dict[str, float] = {source: 0.0}
        pending = {source}
        while pending:
            node = min(pending, key=lambda value: distances[value])
            pending.discard(node)
            for neighbour, weight in (weights.get(node) or {}).items():
                candidate = distances[node] + weight
                if candidate < distances.get(neighbour, float("inf")):
                    distances[neighbour] = candidate
                    pending.add(neighbour)
        return distances

    gaps: List[float] = []
    unreachable = 0
    paths: Dict[str, Dict[str, float]] = {}
    for first in range(len(positions)):
        room_first = camera_rooms[first]
        for second in range(first + 1, len(positions)):
            room_second = camera_rooms[second]
            if room_first not in room_by_id or room_second not in room_by_id:
                unreachable += 1
                continue
            if room_first == room_second:
                gaps.append(math.dist(positions[first], positions[second]))
                continue
            if room_first not in paths:
                paths[room_first] = shortest(room_first)
            through = paths[room_first].get(room_second)
            if through is None:
                unreachable += 1
                continue
            gaps.append(math.dist(positions[first], centroids[room_first]) + through
                        + math.dist(centroids[room_second], positions[second]))
    return {"gaps": gaps, "unreachable": unreachable}


def scene_metrics(spec: dict, seed: Optional[int] = None, layout_report: Optional[dict] = None,
                  path: Optional[str] = None) -> dict:
    """Structure, shape, function, furniture, camera and lighting numbers."""
    graph = scene_graph(spec)
    rooms = graph["rooms"]
    room_by_id = {room["id"]: room for room in rooms}
    index_by_id = {room["id"]: index for index, room in enumerate(rooms)}
    floor_count = len(graph["floor_ids"]) or 1

    edges: List[Tuple[str, str, str]] = []
    for first, second, kind in graph["connections"]:
        if first not in index_by_id or second not in index_by_id or first == second:
            continue
        low, high = sorted((first, second))
        edges.append((low, high, str(kind)))
    edges = sorted(set(edges))
    graph_edges = [(index_by_id[first], index_by_id[second], kind) for first, second, kind in edges]

    kinds: Dict[str, int] = {}
    for _first, _second, kind in edges:
        kinds[kind] = kinds.get(kind, 0) + 1
    cross_floor = sum(1 for first, second, _kind in edges
                      if room_by_id[first]["floor"] != room_by_id[second]["floor"])

    node_attrs = [(room["type"], room["floor"]) for room in rooms]
    structure_key, structure_exact = canonical_graph_key(node_attrs, graph_edges, floor_count)
    cycles = _cycle_rank(len(rooms), graph_edges)

    type_counts: Dict[str, int] = {}
    for room in rooms:
        type_counts[room["type"]] = type_counts.get(room["type"], 0) + 1

    polygons = [room["polygon"] for room in rooms if len(room["polygon"]) >= 3]
    points = [point for polygon in polygons for point in polygon]
    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width, depth = max(xs) - min(xs), max(ys) - min(ys)
        bbox_area = width * depth
        aspect = round(max(width, depth) / max(1e-6, min(width, depth)), 2)
    else:
        width = depth = bbox_area = 0.0
        aspect = 1.0
    footprint = polygon_union_area(polygons)
    floor_areas = {
        floor_label(graph, floor): round(polygon_union_area(
            [room["polygon"] for room in rooms if room["floor"] == floor]), 2)
        for floor in sorted({room["floor"] for room in rooms})
    }
    total_floor_area = sum(floor_areas.values())

    furniture = [item for item in _as_list(spec.get("furniture")) if isinstance(item, dict)]
    furniture_by_room: Dict[str, List[str]] = {}
    for item in furniture:
        furniture_by_room.setdefault(item.get("room", "?"), []).append(str(item.get("asset")))
    combo: Dict[str, List[str]] = {}
    for room in rooms:
        assets = sorted(furniture_by_room.get(room["id"], []))
        combo.setdefault(room["type"], []).extend(assets)
    combo = {room_type_name: sorted(assets) for room_type_name, assets in combo.items()}

    style = spec.get("style") if isinstance(spec.get("style"), dict) else {}
    structure = {
        "floors": floor_count,
        "rooms": type_counts,
        "nodes": [{"type": room["type"], "floor": floor_label(graph, room["floor"])}
                  for room in sorted(rooms, key=lambda room: (room["floor"], room["type"],
                                                              room["id"]))],
        "edges": [{"rooms": [first, second],
                   "types": [room_by_id[first]["type"], room_by_id[second]["type"]],
                   "kind": kind} for first, second, kind in edges],
        "edge_kinds": dict(sorted(kinds.items())),
        "cycles": cycles,
        "exact": structure_exact,
    }
    plan_key = (f"{structure_key}|aspect={_round(aspect, 0.5)}"
                f"|bbox={_round(bbox_area, 10.0)}|floor_area={_round(total_floor_area, 10.0)}")

    return {
        "seed": seed,
        "path": path,
        "family": (layout_report or {}).get("family") or (layout_report or {}).get("shape"),
        "rooms": len(rooms),
        "room_types": type_counts,
        "floors": floor_count,
        "floor_labels": list(graph["floor_ids"]),
        "connections": {"count": len(edges), "kinds": dict(sorted(kinds.items())),
                        "cross_floor": cross_floor},
        "structure": structure,
        "structure_key": structure_key,
        "structure_hash": _hash(structure_key),
        "plan_key": plan_key,
        "plan_hash": _hash(plan_key),
        "shape": {"width_m": round(width, 2), "depth_m": round(depth, 2),
                  "bbox_area_m2": round(bbox_area, 2), "aspect": aspect,
                  "footprint_area_m2": round(footprint, 2),
                  "total_floor_area_m2": round(total_floor_area, 2),
                  "floor_areas_m2": floor_areas,
                  "footprint_ratio": round(footprint / bbox_area, 3) if bbox_area > 0 else None},
        "functions": sorted(type_counts),
        "furniture": {"count": len(furniture),
                      "distinct_assets": len({item.get("asset") for item in furniture}),
                      "combo_hash": _hash(combo), "per_room_type": combo},
        "cameras": _camera_metrics(graph, rooms, room_by_id, edges),
        "lighting": {"sun_elevation_deg": style.get("sun_elevation_deg"),
                     "sun_azimuth_deg": style.get("sun_azimuth_deg"),
                     "sky_strength": style.get("sky_strength"),
                     "exposure": style.get("exposure"),
                     "windows": len(_as_list(spec.get("windows")))},
    }


def _histogram(values: Iterable[float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda entry: float(entry[0])))


def _exact_groups(metrics: Sequence[dict], key_field: str) -> Tuple[Dict[str, List[dict]], int]:
    """Group by the canonical key: hash buckets first, exact key comparison after."""
    buckets: Dict[str, List[dict]] = {}
    for entry in metrics:
        buckets.setdefault(_hash(entry[key_field]), []).append(entry)
    groups: Dict[str, List[dict]] = {}
    collisions = 0
    for bucket in buckets.values():
        split: Dict[str, List[dict]] = {}
        for entry in bucket:
            split.setdefault(entry[key_field], []).append(entry)
        if len(split) > 1:
            collisions += 1
        groups.update(split)
    return groups, collisions


def _scene_min(metrics: Sequence[dict], path: Sequence[str]) -> Optional[float]:
    values = []
    for entry in metrics:
        node = entry
        for step in path:
            node = node[step]
        if node is not None:
            values.append(node)
    return round(min(values), 3) if values else None


def _scene_mean(metrics: Sequence[dict], path: Sequence[str]) -> Optional[float]:
    values = []
    for entry in metrics:
        node = entry
        for step in path:
            node = node[step]
        if node is not None:
            values.append(node)
    return round(statistics.fmean(values), 3) if values else None


def diversity_summary(metrics: Sequence[dict]) -> dict:
    """Aggregate per-scene metrics: histograms, coverage and topology repeats."""
    if not metrics:
        return {"count": 0}

    types: Dict[str, int] = {}
    families: Dict[str, int] = {}
    kind_totals: Dict[str, int] = {}
    for entry in metrics:
        families[str(entry.get("family"))] = families.get(str(entry.get("family")), 0) + 1
        for name in entry["functions"]:
            types[name] = types.get(name, 0) + 1
        for kind, count in entry["connections"]["kinds"].items():
            kind_totals[kind] = kind_totals.get(kind, 0) + count

    groups, structure_collisions = _exact_groups(metrics, "structure_key")
    plan_groups, plan_collisions = _exact_groups(metrics, "plan_key")

    def repeat_report(bucket: Dict[str, List[dict]], collisions: int) -> dict:
        repeated = {key: entries for key, entries in bucket.items() if len(entries) > 1}
        top = sorted(repeated.items(), key=lambda item: -len(item[1]))[:5]
        return {
            "unique": len(bucket),
            "duplicate_rate": round(1.0 - len(bucket) / float(len(metrics)), 4),
            "hash_collisions": collisions,
            "groups": [
                {"hash": _hash(key),
                 "key": key,
                 "count": len(entries),
                 "seeds": sorted([entry["seed"] for entry in entries
                                  if entry.get("seed") is not None])[:20],
                 "families": sorted({str(entry.get("family")) for entry in entries}),
                 "furniture_combos": len({entry["furniture"]["combo_hash"] for entry in entries}),
                 "note": ("same room graph, different furniture"
                          if len({entry["furniture"]["combo_hash"] for entry in entries}) > 1
                          else "same room graph and the same furniture sets")}
                for key, entries in top
            ],
            "repeat_scenes": sum(len(entries) for entries in repeated.values()),
        }

    furniture_metrics = [entry["furniture"] for entry in metrics]
    camera_counts = [entry["cameras"]["count"] for entry in metrics]
    room_counts = [entry["rooms"] for entry in metrics]
    aspects = [entry["shape"]["aspect"] for entry in metrics]
    bbox_areas = [entry["shape"]["bbox_area_m2"] for entry in metrics]
    footprints = [entry["shape"]["footprint_area_m2"] for entry in metrics]
    floor_areas = [entry["shape"]["total_floor_area_m2"] for entry in metrics]
    sun_azimuths = [entry["lighting"]["sun_azimuth_deg"] for entry in metrics
                    if entry["lighting"]["sun_azimuth_deg"] is not None]
    cycles = [entry["structure"]["cycles"] for entry in metrics]
    return {
        "count": len(metrics),
        "families": dict(sorted(families.items())),
        "rooms": {"histogram": _histogram(room_counts), "min": min(room_counts),
                  "max": max(room_counts), "mean": round(statistics.fmean(room_counts), 2)},
        "floors": _histogram([entry["floors"] for entry in metrics]),
        "cycles": {"histogram": _histogram(cycles), "max": max(cycles),
                   "loops": sum(1 for value in cycles if value > 0)},
        "connections": {
            "total": sum(entry["connections"]["count"] for entry in metrics),
            "kinds": dict(sorted(kind_totals.items())),
            "cross_floor": sum(entry["connections"]["cross_floor"] for entry in metrics),
            "scenes_with_stairs": sum(1 for entry in metrics
                                      if entry["connections"]["kinds"].get("stair")),
            "scenes_with_ramps": sum(1 for entry in metrics
                                     if entry["connections"]["kinds"].get("ramp")),
            "scenes_without_connection": sum(1 for entry in metrics
                                             if not entry["connections"]["count"]),
        },
        "function_coverage": dict(sorted(types.items(), key=lambda entry: -entry[1])),
        "shape": {"aspect_min": min(aspects), "aspect_max": max(aspects),
                  "aspect_mean": round(statistics.fmean(aspects), 2),
                  "bbox_area_min_m2": min(bbox_areas), "bbox_area_max_m2": max(bbox_areas),
                  "bbox_area_mean_m2": round(statistics.fmean(bbox_areas), 2),
                  "footprint_area_min_m2": min(footprints),
                  "footprint_area_max_m2": max(footprints),
                  "footprint_area_mean_m2": round(statistics.fmean(footprints), 2),
                  "total_floor_area_min_m2": min(floor_areas),
                  "total_floor_area_max_m2": max(floor_areas),
                  "total_floor_area_mean_m2": round(statistics.fmean(floor_areas), 2)},
        "furniture": {
            "count_min": min(entry["count"] for entry in furniture_metrics),
            "count_max": max(entry["count"] for entry in furniture_metrics),
            "count_mean": round(statistics.fmean([entry["count"] for entry in furniture_metrics]), 2),
            "distinct_assets_mean": round(statistics.fmean(
                [entry["distinct_assets"] for entry in furniture_metrics]), 2),
        },
        "cameras": {"count_min": min(camera_counts), "count_max": max(camera_counts),
                    "count_mean": round(statistics.fmean(camera_counts), 2),
                    "rooms_without_camera": sum(len(entry["cameras"]["rooms_without_camera"])
                                                for entry in metrics),
                    "min_gap_3d_m_min": _scene_min(metrics, ("cameras", "min_gap_3d_m")),
                    "mean_gap_3d_m_mean": _scene_mean(metrics, ("cameras", "mean_gap_3d_m")),
                    "min_gap_same_floor_m_min": _scene_min(
                        metrics, ("cameras", "min_gap_same_floor_m")),
                    "mean_gap_same_floor_m_mean": _scene_mean(
                        metrics, ("cameras", "mean_gap_same_floor_m")),
                    "min_gap_nav_m_min": _scene_min(metrics, ("cameras", "min_gap_nav_m")),
                    "mean_gap_nav_m_mean": _scene_mean(metrics, ("cameras", "mean_gap_nav_m")),
                    "unreachable_pairs": sum(entry["cameras"]["unreachable_pairs"]
                                             for entry in metrics)},
        "lighting": {"sun_azimuth_distinct": len(set(sun_azimuths)),
                     "sun_azimuth_values": sorted(set(sun_azimuths))[:12]},
        "structures": repeat_report(groups, structure_collisions),
        "plans": repeat_report(plan_groups, plan_collisions),
    }


# --------------------------------------------------------------------------- #
# final products: the request, the build and what survived the pipeline
#
# A diversity number over the *successful* scenes answers a different question
# than one over the generated specs: the pipeline filters (a render that failed,
# a review that failed) and the generator falls back (a family that could not be
# built is replaced, or re-drawn per attempt), so the delivered mix can be much
# narrower than the requested one.  This section reads a batch root --
# ``batch_summary.json`` plus each scene's ``spec.json`` / ``layout_report.json``
# / ``run_summary.json`` -- and reports the request, the build and the surviving
# products side by side, with the fallbacks and the filtering that moved one into
# the other.  It never counts a generation request as a product.
# --------------------------------------------------------------------------- #
def _layout_module():
    """``blender_agent.layout_gen`` when importable (it is pure python, no bpy)."""
    try:
        from blender_agent import layout_gen as layout_mod  # noqa: PLC0415 - optional import
    except Exception:
        return None
    return layout_mod


def draw_family(seed: int, rooms: Optional[int] = None) -> Optional[dict]:
    """The ``{"rooms", "family"}`` ``--shape auto`` draws for one seed.

    The draw runs on the generator's own stream, so this reproduces what
    ``layout_gen.py --seed <seed> --shape auto`` asks for.  It is only used for
    artifacts that predate ``stats.auto_pick``: without it a legacy batch reads
    as "every scene asked for the family it happened to build", which is exactly
    the bias the report is supposed to expose.
    """
    module = _layout_module()
    if module is None:
        return None
    try:
        count, family = module._pick_auto_shape(
            random.Random(seed * ATTEMPT_SALT + module.AUTO_PICK_SALT), rooms)
    except Exception:
        return None
    return {"rooms": int(count), "family": str(family)}


def auto_weight_share() -> Optional[Dict[str, float]]:
    """The generator's draw weights, normalized: the *a priori* ``auto`` mix."""
    module = _layout_module()
    weights = getattr(module, "AUTO_FAMILY_WEIGHT", None) if module else None
    if not isinstance(weights, dict) or not weights:
        return None
    return _shares({str(name): float(value) for name, value in weights.items()}) or None


def _read_json(path: Optional[str]):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def scene_facts(scene_dir: str) -> dict:
    """The three per-scene files a success report reads, plus where they are."""
    paths = {name: os.path.join(scene_dir, name)
             for name in ("spec.json", "layout_report.json", "run_summary.json")}
    return {"dir": scene_dir, "paths": paths,
            "spec": _read_json(paths["spec.json"]),
            "layout_report": _read_json(paths["layout_report.json"]),
            "run_summary": _read_json(paths["run_summary.json"])}


def scene_family(facts: dict) -> dict:
    """What one scene asked for -- as far as its artifacts record it -- and built.

    ``layout_report.json`` and ``run_summary.json`` carry the same ``stats``
    block, so either answers when the other is missing.  ``stats.auto_pick`` and
    ``stats.auto_fallback`` are the contract fields of the current generator; a
    report that predates them yields ``None`` here instead of a made-up "no
    fallback".
    """
    layout = facts.get("layout_report") or {}
    summary = facts.get("run_summary") or {}
    stats = layout.get("stats")
    if not isinstance(stats, dict) or not stats:
        stats = summary.get("stats") if isinstance(summary.get("stats"), dict) else {}
    # the *report* carries the arrangement the attempt built ("shape" of the
    # attempt, next to "family" for the topology families); stats.shape is the
    # same value on the current generator but the *requested* shape on a report
    # that predates stats.family, so it comes last
    requested_shape = stats.get("shape") or layout.get("shape") or summary.get("shape")
    built = (stats.get("family") or layout.get("family")
             or layout.get("shape") or stats.get("shape") or summary.get("shape"))
    recorded = None
    auto_pick = stats.get("auto_pick")
    if isinstance(auto_pick, (list, tuple)) and auto_pick:
        recorded = {"rooms": int(auto_pick[0]) if len(auto_pick) > 0 else None,
                    "family": str(auto_pick[1]) if len(auto_pick) > 1 else None}
    attempts = stats.get("attempts_used")
    if attempts is None and stats.get("attempt") is not None:
        attempts = int(stats["attempt"]) + 1
    fallback = stats.get("auto_fallback")
    return {"built": str(built) if built else None,
            "requested_shape": str(requested_shape) if requested_shape else None,
            "recorded_draw": recorded,
            "attempts": None if attempts is None else int(attempts),
            "auto_fallback": None if fallback is None else bool(fallback),
            "floors": stats.get("floors")}


def _requested_family(family: dict, root_block: dict, seed: int,
                      replay: bool) -> Tuple[Optional[str], str]:
    """The family a scene was asked for, and how that was established.

    ``recorded`` is the generator's own ``stats.auto_pick``, ``declared`` a
    request that names a family (or the duplex plan, which ``--floors 2``
    forces), ``replayed`` a re-draw with the current generator for an artifact
    that records nothing, ``unknown`` neither.
    """
    drawn = (family.get("recorded_draw") or {}).get("family")
    if drawn:
        return str(drawn), "recorded"
    # the batch request names the shape; a legacy layout report recorded it in
    # stats.shape, which the current generator overwrites with the built one
    shape = root_block.get("shape") or family.get("requested_shape")
    if family.get("floors") == 2 or shape == "duplex":
        return "duplex", "declared"
    module = _layout_module()
    families = set(getattr(module, "ROOM_FAMILIES", {}) or {}) if module else set()
    if shape and (shape in families or shape in ("row", "block")):
        return str(shape), "declared"
    if replay and shape == "auto":
        replayed = draw_family(seed, root_block.get("rooms"))
        if replayed is not None:
            return str(replayed["family"]), "replayed"
    return None, "unknown"


def _coverage_facts(facts: dict) -> dict:
    """How much of its room each camera of one scene sees.

    ``door_visual_forecast`` (written by layout_gen and copied into the run
    summary) is the only record of the walkable floor a stand covers; a scene
    without it reports ``None`` rather than a zero that would read as a blind
    camera.
    """
    forecast = (facts.get("layout_report") or {}).get("door_visual_forecast")
    if not isinstance(forecast, dict) or not forecast:
        forecast = (facts.get("run_summary") or {}).get("door_visual_forecast")
    areas: List[float] = []
    doors: List[float] = []
    if isinstance(forecast, dict):
        for entry in forecast.values():
            if not isinstance(entry, dict):
                continue
            coverage = entry.get("coverage")
            value = coverage.get("visible_area_fraction") if isinstance(coverage, dict) else None
            if isinstance(value, (int, float)):
                areas.append(float(value))
            fraction = entry.get("visible_fraction")
            if isinstance(fraction, (int, float)):
                doors.append(float(fraction))
    return {"views": len(areas),
            "visible_area_fraction_min": round(min(areas), 3) if areas else None,
            "visible_area_fraction_mean": round(statistics.fmean(areas), 3) if areas else None,
            "visible_area_fraction_max": round(max(areas), 3) if areas else None,
            "views_below_half": sum(1 for value in areas if value < 0.5),
            "door_visible_fraction_mean": round(statistics.fmean(doors), 3) if doors else None}


def _quality_facts(facts: dict) -> dict:
    """What a scene's own run summary says about how good it turned out."""
    summary = facts.get("run_summary") or {}
    review = summary.get("review") if isinstance(summary.get("review"), dict) else {}
    validation = summary.get("validation") if isinstance(summary.get("validation"), dict) else {}
    render = summary.get("render") if isinstance(summary.get("render"), dict) else {}
    ok = summary.get("ok") if isinstance(summary.get("ok"), dict) else {}
    return {"review_pass": review.get("pass"),
            "review_score": review.get("score"),
            "review_failed_checks": list(review.get("failed_checks") or []),
            "review_ok": ok.get("review_ok"),
            "validation_pass": validation.get("pass"),
            "validation_pass_at_resolution": validation.get("pass_at_resolution"),
            "render_status": render.get("status")}


def quality_stats(records: Sequence[dict]) -> dict:
    """The quality of a set next to its diversity: review, validation, profile.

    Diversity alone is half an answer -- a mix that grew more varied by shipping
    worse houses is not a better dataset -- so the review verdicts, the
    validation result and the effective render profile of the very scenes the
    mix was measured on are reported family by family.
    """
    reviewed = review_passed = validated = at_resolution = degraded = 0
    per_family: Dict[str, dict] = {}
    failed_checks: Dict[str, int] = {}
    for record in records:
        quality = record.get("quality") or {}
        family = str(record.get("family") or "unknown")
        bucket = per_family.setdefault(family, {"scenes": 0, "reviewed": 0, "review_pass": 0,
                                                "scores": []})
        bucket["scenes"] += 1
        if quality.get("review_pass") is not None:
            reviewed += 1
            bucket["reviewed"] += 1
            if quality["review_pass"]:
                review_passed += 1
                bucket["review_pass"] += 1
        if isinstance(quality.get("review_score"), (int, float)):
            bucket["scores"].append(float(quality["review_score"]))
        if quality.get("validation_pass") is not None:
            validated += 1 if quality["validation_pass"] else 0
        if quality.get("validation_pass_at_resolution") is not None:
            at_resolution += 1 if quality["validation_pass_at_resolution"] else 0
        for name in quality.get("review_failed_checks") or []:
            failed_checks[str(name)] = failed_checks.get(str(name), 0) + 1
        if (record.get("render_quality") or {}).get("degraded"):
            degraded += 1
    scene_scores = [score for bucket in per_family.values() for score in bucket["scores"]]
    return {
        "scenes": len(records),
        "reviewed": reviewed,
        "review_passed": review_passed,
        "review_score": {"mean": round(statistics.fmean(scene_scores), 3) if scene_scores else None,
                         "min": min(scene_scores) if scene_scores else None,
                         "max": max(scene_scores) if scene_scores else None},
        "validation_passed": validated,
        "validation_passed_at_resolution": at_resolution,
        "degraded": degraded,
        "review_failed_checks": dict(sorted(failed_checks.items(), key=lambda item: (-item[1],
                                                                                    item[0]))),
        "by_family": {family: {"scenes": bucket["scenes"], "reviewed": bucket["reviewed"],
                               "review_pass": bucket["review_pass"],
                               "review_score_mean": round(statistics.fmean(bucket["scores"]), 3)
                               if bucket["scores"] else None}
                      for family, bucket in sorted(per_family.items())},
    }


def _seed_of(row: dict, scene_dir: str) -> Optional[int]:
    """The seed of a batch row: recorded, else the ``scene_<seed>`` directory."""
    seed = row.get("seed")
    if seed is not None:
        try:
            return int(seed)
        except (TypeError, ValueError):
            return None
    name = os.path.basename(str(scene_dir).rstrip("/"))
    if name.startswith("scene_"):
        try:
            return int(name[len("scene_"):])
        except ValueError:
            return None
    return None


def _summary_row(scene_dir: str, summary: dict) -> dict:
    """A batch-shaped row for a scene directory that has no batch summary."""
    ok = summary.get("ok") if isinstance(summary.get("ok"), dict) else None
    if ok is None:
        layout_ok = bool((summary.get("checks") or {}).get("pass"))
        ok = {"layout_ok": layout_ok, "render_ok": None, "validation_ok": None,
              "bundle_ok": layout_ok}
    status = "bundle_ok" if ok.get("bundle_ok") else "failed"
    return {"seed": summary.get("seed"), "dir": os.path.basename(scene_dir.rstrip("/")),
            "status": status, "ok": ok, "failure": summary.get("failure"),
            "render_quality": None, "render_requested": summary.get("render_requested")}


def read_root(root: str, replay_draw: bool = True) -> dict:
    """One batch root: its ``batch_summary.json``, its rows and its records.

    A root without a ``batch_summary.json`` still works -- every ``scene_*``
    directory is read through its own ``run_summary.json``, which carries the
    same grades and failure record (older output directories have no batch
    summary).  The request the root was bound to is taken from the batch
    summary's ``request`` block, or from a benchmark ``report.json`` next to it.
    """
    root = os.path.abspath(root)
    if os.path.isfile(root) and os.path.basename(root) == "batch_summary.json":
        root = os.path.dirname(root)
    summary = _read_json(os.path.join(root, "batch_summary.json"))
    if isinstance(summary, dict):
        block = {
            "root": root,
            "batch_id": summary.get("batch_id"),
            "levels": summary.get("levels"),
            "passed": summary.get("passed"),
            "count": summary.get("count"),
            "request": summary.get("request") if isinstance(summary.get("request"), dict) else None,
            "render_requested": bool(summary.get("render_requested")),
            "review_requested": bool(summary.get("review_requested")),
            "rows": [row for row in _as_list(summary.get("scenes")) if isinstance(row, dict)],
            "source": "batch_summary",
        }
    else:
        rows = []
        for path in sorted(glob.glob(os.path.join(root, "scene_*"))):
            if not os.path.isdir(path):
                continue
            summary_of_scene = _read_json(os.path.join(path, "run_summary.json"))
            if isinstance(summary_of_scene, dict):
                rows.append(_summary_row(path, summary_of_scene))
        if not rows:
            raise ValueError("no batch_summary.json and no scene_*/run_summary.json under %s" % root)
        block = {"root": root, "batch_id": None, "levels": None, "passed": None,
                 "count": len(rows), "request": None, "rows": rows,
                 "render_requested": any(row.get("render_requested") for row in rows),
                 "review_requested": any("review_ok" in (row.get("ok") or {}) for row in rows),
                 "source": "scene_dirs"}

    report = _read_json(os.path.join(root, "report.json"))
    config = (report or {}).get("config") if isinstance(report, dict) else None
    request = block["request"] or {}
    layout = request.get("layout") if isinstance(request.get("layout"), dict) else {}
    if isinstance(config, dict):
        layout = dict(layout, **{key: layout.get(key) or config.get(key)
                                 for key in ("shape", "rooms")})
    block["shape"] = layout.get("shape")
    block["rooms"] = int(layout["rooms"]) if str(layout.get("rooms") or "").isdigit() else None
    block["role"] = None
    block["records"] = [scene_record(block, row, replay_draw) for row in block["rows"]
                        if _seed_of(row, os.path.join(root, str(row.get("dir") or ""))) is not None]
    block["skipped_rows"] = len(block["rows"]) - len(block["records"])
    return block


def scene_record(root_block: dict, row: dict, replay_draw: bool = True) -> dict:
    """One scene of one batch: grades, family facts, spec metrics and coverage."""
    scene_dir = os.path.join(root_block["root"], str(row.get("dir") or ""))
    seed = _seed_of(row, scene_dir)
    facts = scene_facts(scene_dir)
    family = scene_family(facts)
    spec = facts["spec"]
    metrics = None
    if isinstance(spec, dict):
        try:
            metrics = scene_metrics(spec, seed=seed, layout_report=facts["layout_report"],
                                    path=facts["paths"]["spec.json"])
        except Exception:
            metrics = None
    if metrics is not None:
        # the family the *report* names wins over the reader's tolerant guess
        # (a scene without a layout report falls back to its own shape); a scene
        # that records neither is counted as unknown, never as a family named
        # "None"
        metrics["family"] = family["built"] or metrics.get("family") or "unknown"
    requested, basis = _requested_family(family, root_block, seed, replay_draw)
    cameras = (metrics or {}).get("cameras") or {}
    return {
        "seed": seed,
        "root": root_block["root"],
        "role": root_block.get("role"),
        "dir": row.get("dir"),
        "status": row.get("status"),
        "ok": row.get("ok") or {},
        "failure": row.get("failure"),
        "render_quality": row.get("render_quality"),
        "family": family["built"],
        "requested_shape": root_block.get("shape") or family["requested_shape"],
        "requested_family": requested,
        "requested_family_basis": basis,
        "requested_rooms": (family["recorded_draw"] or {}).get("rooms")
        or root_block.get("rooms"),
        "attempts": family["attempts"],
        "auto_fallback": family["auto_fallback"],
        "floors": (metrics or {}).get("floors") or family["floors"] or row.get("floors"),
        "rooms": (metrics or {}).get("rooms") or len(row.get("rooms") or []) or None,
        "cameras": cameras.get("count") or row.get("cameras"),
        "furniture": ((metrics or {}).get("furniture") or {}).get("count")
        or row.get("furniture"),
        "quality": _quality_facts(facts),
        "coverage": _coverage_facts(facts),
        "metrics": metrics,
        "spec": spec,
    }


def _set_role(blocks: Sequence[dict], product_roots: Optional[Sequence[str]]) -> List[dict]:
    """Mark which roots define the final products.

    Explicit ``product_roots`` win.  Otherwise a root that asked for a render is
    a product batch and a layout-only root is not; when no root asked for one
    (a layout sweep) every root is a product batch, because bundle_ok *is*
    layout_ok there.
    """
    named = {os.path.abspath(path) for path in (product_roots or [])}
    rendered = any(block["render_requested"] for block in blocks)
    for block in blocks:
        if named:
            block["role"] = "product" if block["root"] in named else "generation"
        else:
            block["role"] = "product" if (not rendered
                                           or block["render_requested"]) else "generation"
        # the records were built before the roles were known; `_merge_by_seed`
        # prefers the product record of a seed, so they carry it too
        for record in block["records"]:
            record["role"] = block["role"]
    return blocks


def _merge_by_seed(blocks: Sequence[dict]) -> List[dict]:
    """One record per seed, the batch that delivered the most of that seed first.

    A seed can sit in two batches (a layout sweep and the rendered sample): the
    generated set takes the spec that passed the layout stage, the product set
    the run that actually produced artifacts.
    """
    def rank(record: dict) -> tuple:
        ok = record.get("ok") or {}
        return (bool(ok.get("layout_ok")), record.get("role") == "product",
                record.get("metrics") is not None)

    best: Dict[int, dict] = {}
    for block in blocks:
        for record in block["records"]:
            seed = record["seed"]
            if seed not in best or rank(record) > rank(best[seed]):
                best[seed] = record
    return [best[seed] for seed in sorted(best)]


def _shares(counts: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {str(key): round(float(value) / total, 4) for key, value in sorted(counts.items())}


def _tv_distance(left: Dict[str, float], right: Dict[str, float]) -> float:
    """Total variation of two mixes: half the sum of the share differences."""
    keys = set(left) | set(right)
    return round(0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys), 4)


def _bin_label(low: float, high: Optional[float]) -> str:
    return "%.1f-%.1f" % (low, high) if high is not None else ">=%.1f" % low


def _bin_counts(values: Sequence[float], edges: Sequence[float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for low, high in zip(edges, list(edges[1:]) + [None]):
        out[_bin_label(low, high)] = sum(1 for value in values
                                         if value >= low and (high is None or value < high))
    return out


def shape_bins(metrics: Sequence[dict]) -> dict:
    """Aspect and area bins next to the means: a 4:1 railroad is not a block."""
    return {"aspect": _bin_counts([entry["shape"]["aspect"] for entry in metrics], ASPECT_BINS),
            "footprint_area_m2": _bin_counts(
                [entry["shape"]["footprint_area_m2"] for entry in metrics], AREA_BINS),
            "total_floor_area_m2": _bin_counts(
                [entry["shape"]["total_floor_area_m2"] for entry in metrics], AREA_BINS)}


def door_graph(metrics: Sequence[dict], summary: Optional[dict] = None) -> dict:
    """The connection graph of a set: kinds, loops, room degrees and repeats.

    The degree histogram says *how* connected the set is -- a house of chain
    rooms is mostly leaves (degree 1) and one hub, a loop plan is not -- and the
    topology repeats say how much of it is the same room graph with other rooms.
    """
    kinds: Dict[str, int] = {}
    degrees: Dict[int, int] = {}
    cycles: List[int] = []
    rooms = edges_total = cross_floor = without = 0
    for entry in metrics:
        structure = entry.get("structure") or {}
        connections = entry.get("connections") or {}
        rooms += int(entry.get("rooms") or 0)
        edges_total += int(connections.get("count") or 0)
        cross_floor += int(connections.get("cross_floor") or 0)
        without += 0 if connections.get("count") else 1
        for kind, count in (structure.get("edge_kinds") or {}).items():
            kinds[str(kind)] = kinds.get(str(kind), 0) + int(count)
        degree: Dict[str, int] = {}
        for edge in structure.get("edges") or []:
            for node in edge.get("rooms") or []:
                degree[str(node)] = degree.get(str(node), 0) + 1
        for value in list(degree.values()) + [0] * max(0, int(entry.get("rooms") or 0)
                                                       - len(degree)):
            degrees[value] = degrees.get(value, 0) + 1
        cycles.append(int(structure.get("cycles") or 0))
    structures = (summary or {}).get("structures") or {}
    plans = (summary or {}).get("plans") or {}
    return {
        "rooms": rooms,
        "edges": edges_total,
        "kinds": dict(sorted(kinds.items())),
        "cross_floor": cross_floor,
        "cycles": _histogram(cycles),
        "scenes_with_cycles": sum(1 for value in cycles if value > 0),
        "scenes_without_connection": without,
        "degree_histogram": {str(degree): count for degree, count in sorted(degrees.items())},
        "leaf_rooms": degrees.get(1, 0),
        "leaf_share": round(degrees.get(1, 0) / rooms, 4) if rooms else None,
        "topologies": {"unique": structures.get("unique"),
                       "duplicate_rate": structures.get("duplicate_rate")},
        "plans": {"unique": plans.get("unique"), "duplicate_rate": plans.get("duplicate_rate")},
    }


def fallback_stats(records: Sequence[dict]) -> dict:
    """How many products are the *first* attempt, and how many fell back.

    Three fallbacks stay apart because the artifacts record them differently:
    a **retry** (``attempts_used > 1``: the first attempt was rejected), the
    generator's **auto fallback** (the sticky ``--shape auto`` draw gave up and
    re-drew per attempt) and a **family switch** (the drawn family is not the
    built one -- a failure *and* a fallback).  ``unknown`` counts the scenes
    whose report predates a field, so a legacy batch cannot read as "nothing
    fell back".

    The headline family-switch number is over the **recorded** draws only: a
    re-drawn (``replayed``) family is a *today's* request for a legacy artifact,
    not the request it was built against, so it stays in ``by_basis`` next to
    the real one instead of inflating the rate.
    """
    attempts: Dict[str, int] = {}
    switch_pairs: Dict[str, int] = {}
    by_basis: Dict[str, Dict[str, int]] = {}
    retried = fallback_known = fallback_true = 0
    unknown_attempts = unknown_fallback = unknown_draw = 0
    for record in records:
        used = record.get("attempts")
        if used is None:
            unknown_attempts += 1
        else:
            attempts[str(int(used))] = attempts.get(str(int(used)), 0) + 1
            if int(used) > 1:
                retried += 1
        if record.get("auto_fallback") is None:
            unknown_fallback += 1
        else:
            fallback_known += 1
            fallback_true += 1 if record["auto_fallback"] else 0
        drawn, built = record.get("requested_family"), record.get("family")
        if drawn is None or built is None:
            unknown_draw += 1
            continue
        basis = str(record.get("requested_family_basis") or "unknown")
        bucket = by_basis.setdefault(basis, {"known": 0, "switched": 0})
        bucket["known"] += 1
        if str(drawn) != str(built):
            bucket["switched"] += 1
            if basis == "recorded":
                pair = "%s->%s" % (drawn, built)
                switch_pairs[pair] = switch_pairs.get(pair, 0) + 1
    total = len(records)
    first = attempts.get("1", 0)
    recorded = by_basis.get("recorded", {"known": 0, "switched": 0})

    def rate(value: int, denominator: int) -> Optional[float]:
        return round(value / denominator, 4) if denominator else None

    return {
        "scenes": total,
        "attempt_histogram": dict(sorted(attempts.items(), key=lambda item: int(item[0]))),
        "first_attempt": first,
        "first_attempt_rate": rate(first, total),
        "retried": retried,
        "retry_rate": rate(retried, total),
        "auto_fallback": {"known": fallback_known, "true": fallback_true,
                          "rate": rate(fallback_true, fallback_known),
                          "unknown": unknown_fallback},
        "family_switch": {"known": recorded["known"], "switched": recorded["switched"],
                          "rate": rate(recorded["switched"], recorded["known"]),
                          "unknown": unknown_draw,
                          "by_basis": {basis: dict(bucket, rate=rate(bucket["switched"],
                                                                     bucket["known"]))
                                       for basis, bucket in sorted(by_basis.items())},
                          "pairs": dict(sorted(switch_pairs.items(),
                                               key=lambda item: (-item[1], item[0])))},
        "unknown": {"attempts": unknown_attempts, "auto_fallback": unknown_fallback,
                    "draw": unknown_draw},
    }


def asset_reuse(records: Sequence[dict]) -> dict:
    """How much of the furniture library the products reuse.

    ``reuse_rate`` is ``1 - distinct / placements``: 0 when every placed item is
    a different asset, high when a handful of assets furnish every scene.  The
    cross-scene numbers say whether the reuse is *within* a scene (the same
    wardrobe twice) or across the whole set (the same sofa in every house).
    """
    placements = 0
    per_asset: Dict[str, int] = {}
    scenes_of: Dict[str, set] = {}
    by_prefix: Dict[str, Dict[str, int]] = {}
    distinct_per_scene: List[int] = []
    scene_count = 0
    for record in records:
        spec = record.get("spec")
        if not isinstance(spec, dict):
            continue
        scene_count += 1
        assets = [str(item.get("asset")) for item in _as_list(spec.get("furniture"))
                  if isinstance(item, dict)]
        placements += len(assets)
        for asset in assets:
            per_asset[asset] = per_asset.get(asset, 0) + 1
            scenes_of.setdefault(asset, set()).add(record.get("seed"))
            prefix = asset.split(":", 1)[0] if ":" in asset else "other"
            bucket = by_prefix.setdefault(prefix, {"placements": 0, "distinct": 0})
            bucket["placements"] += 1
        distinct_per_scene.append(len(set(assets)))
    for asset, count in per_asset.items():
        prefix = asset.split(":", 1)[0] if ":" in asset else "other"
        by_prefix[prefix]["distinct"] += 1
    shared = sum(1 for asset in per_asset if len(scenes_of[asset]) > 1)
    top = sorted(per_asset.items(), key=lambda item: (-item[1], item[0]))[:12]
    return {
        "scenes": scene_count,
        "placements": placements,
        "distinct_assets": len(per_asset),
        "reuse_rate": round(1.0 - len(per_asset) / placements, 4) if placements else None,
        "assets_in_one_scene": sum(1 for asset in per_asset if len(scenes_of[asset]) == 1),
        "assets_in_many_scenes": shared,
        "cross_scene_reuse_rate": round(shared / len(per_asset), 4) if per_asset else None,
        "distinct_per_scene_mean": round(statistics.fmean(distinct_per_scene), 2)
        if distinct_per_scene else None,
        "items_per_scene_mean": round(placements / scene_count, 2) if scene_count else None,
        "by_prefix": {prefix: dict(bucket, placements=bucket["placements"])
                      for prefix, bucket in sorted(by_prefix.items())},
        "top": [{"asset": asset, "placements": count, "scenes": len(scenes_of[asset])}
                for asset, count in top],
    }


def view_coverage(records: Sequence[dict]) -> dict:
    """View coverage of a set: stands per room, rooms left without one, what they see.

    ``room_coverage`` counts a room as covered when it has at least one camera;
    transit rooms (corridor / balcony / entry) are exempt in the per-scene
    ``rooms_without_camera`` list, exactly as the metric itself defines it.
    ``visible_area_fraction`` is the share of the walkable floor a stand sees.
    """
    cameras_per_scene: List[int] = []
    doors: List[float] = []
    scene_means: List[float] = []
    views = below_half = rooms = missing_rooms = covered_rooms = scenes_without_forecast = 0
    for record in records:
        metrics = record.get("metrics") or {}
        cameras = metrics.get("cameras") or {}
        if cameras.get("count") is not None:
            cameras_per_scene.append(int(cameras["count"]))
        rooms += int(metrics.get("rooms") or 0)
        missing_rooms += len(cameras.get("rooms_without_camera") or [])
        covered_rooms += len(cameras.get("per_room") or {})
        coverage = record.get("coverage") or {}
        views += int(coverage.get("views") or 0)
        below_half += int(coverage.get("views_below_half") or 0)
        if coverage.get("views"):
            scene_means.append(float(coverage["visible_area_fraction_mean"]))
        else:
            scenes_without_forecast += 1
        if coverage.get("door_visible_fraction_mean") is not None:
            doors.append(float(coverage["door_visible_fraction_mean"]))
    return {
        "scenes": len(records),
        "cameras": sum(cameras_per_scene),
        "cameras_per_scene": {"min": min(cameras_per_scene) if cameras_per_scene else None,
                              "mean": round(statistics.fmean(cameras_per_scene), 2)
                              if cameras_per_scene else None,
                              "max": max(cameras_per_scene) if cameras_per_scene else None,
                              "histogram": _histogram(cameras_per_scene)},
        "rooms": rooms,
        "rooms_with_camera": covered_rooms,
        "rooms_without_camera": missing_rooms,
        "room_coverage": round(covered_rooms / rooms, 4) if rooms else None,
        "views_with_forecast": views,
        "views_below_half": below_half,
        "views_below_half_rate": round(below_half / views, 4) if views else None,
        "scenes_without_forecast": scenes_without_forecast,
        "scene_visible_area_fraction": {"min": min(scene_means) if scene_means else None,
                                        "mean": round(statistics.fmean(scene_means), 3)
                                        if scene_means else None,
                                        "max": max(scene_means) if scene_means else None},
        "door_visible_fraction_mean": round(statistics.fmean(doors), 3) if doors else None,
    }


def requested_stats(records: Sequence[dict]) -> dict:
    """What a set of scenes *asked for*: the drawn family, floors and rooms."""
    basis: Dict[str, int] = {}
    families: Dict[str, int] = {}
    floors: Dict[str, int] = {}
    rooms: Dict[str, int] = {}
    shapes: Dict[str, int] = {}
    for record in records:
        basis[str(record.get("requested_family_basis"))] = \
            basis.get(str(record.get("requested_family_basis")), 0) + 1
        if record.get("requested_family"):
            families[str(record["requested_family"])] = \
                families.get(str(record["requested_family"]), 0) + 1
        if record.get("floors") is not None:
            floors[str(record["floors"])] = floors.get(str(record["floors"]), 0) + 1
        if record.get("requested_rooms"):
            rooms[str(record["requested_rooms"])] = rooms.get(str(record["requested_rooms"]), 0) + 1
        if record.get("requested_shape"):
            shapes[str(record["requested_shape"])] = \
                shapes.get(str(record["requested_shape"]), 0) + 1
    return {"scenes": len(records),
            "basis": dict(sorted(basis.items())),
            "families": dict(sorted(families.items())),
            "share": _shares(families),
            "unknown_family": sum(1 for record in records if not record.get("requested_family")),
            "requested_shapes": dict(sorted(shapes.items())),
            "floors": dict(sorted(floors.items())),
            "rooms": dict(sorted(rooms.items())),
            "auto_weight_share": auto_weight_share()}


def _survival(requested: Sequence[dict], actual: Sequence[dict]) -> Dict[str, dict]:
    """Of the seeds that asked for a family, how many delivered *that* family.

    Counting the delivered set by family and dividing by the requested count is
    not a survival rate: a fallback moves a seed from the family it asked for to
    another one, so the family it lands in can be built more often than it was
    requested (``built`` > ``asked``).  Joining the two sets by seed is what the
    rate has to mean.
    """
    built_by_seed = {record["seed"]: record.get("family") for record in actual}
    out: Dict[str, dict] = {}
    for record in requested:
        drawn = record.get("requested_family")
        if drawn is None:
            continue
        bucket = out.setdefault(str(drawn), {"asked": 0, "kept": 0})
        bucket["asked"] += 1
        if built_by_seed.get(record["seed"]) is not None \
                and str(built_by_seed[record["seed"]]) == str(drawn):
            bucket["kept"] += 1
    return out


def _set_block(records: Sequence[dict]) -> dict:
    """Every statistic a set of final products is graded on, in one block."""
    metrics = [record["metrics"] for record in records if record.get("metrics")]
    summary = diversity_summary(metrics)
    summary["family_share"] = _shares(summary.get("families") or {})
    summary["shape_bins"] = shape_bins(metrics)
    summary["door_graph"] = door_graph(metrics, summary)
    summary["fallback"] = fallback_stats(records)
    summary["furniture_reuse"] = asset_reuse(records)
    summary["view_coverage"] = view_coverage(records)
    summary["quality"] = quality_stats(records)
    return summary


def _filtered_block(product_records: Sequence[dict], success: Sequence[dict],
                    requested: Sequence[dict]) -> dict:
    """What the pipeline dropped between the request and the delivered set."""
    kept = {record["seed"] for record in success}
    dropped = [record for record in product_records if record["seed"] not in kept]
    reasons: Dict[str, int] = {}
    families: Dict[str, int] = {}
    for record in dropped:
        failure = record.get("failure") or {}
        key = "%s:%s" % (failure.get("stage", "unknown"), failure.get("kind", "not_bundle_ok"))
        reasons[key] = reasons.get(key, 0) + 1
        families[str(record.get("family"))] = families.get(str(record.get("family")), 0) + 1
    return {"requested": len(product_records),
            "generated_requested": len({record["seed"] for record in requested}),
            "products": len(product_records),
            "kept": len(success),
            "dropped": len(dropped),
            "drop_rate": round(len(dropped) / len(product_records), 4) if product_records else None,
            "reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
            "dropped_families": dict(sorted(families.items())),
            "seeds": sorted(record["seed"] for record in dropped)}


def _bias_notes(requested: dict, actual: dict, label: str) -> List[str]:
    """Sentences a reader can act on: what moved, what vanished, what is unknown."""
    notes: List[str] = []
    req = requested.get("share") or {}
    act = actual.get("family_share") or {}
    for family in sorted(set(req) | set(act)):
        if req.get(family, 0.0) - act.get(family, 0.0) > 0.05:
            notes.append("%s: '%s' loses %.0f%% -> %.0f%% of the mix"
                         % (label, family, 100 * req.get(family, 0.0),
                            100 * act.get(family, 0.0)))
        elif act.get(family, 0.0) - req.get(family, 0.0) > 0.05:
            notes.append("%s: '%s' grows %.0f%% -> %.0f%% of the mix"
                         % (label, family, 100 * req.get(family, 0.0),
                            100 * act.get(family, 0.0)))
    lost = [family for family in sorted(req) if not (actual.get("families") or {}).get(family)]
    if lost:
        notes.append("%s: requested but never delivered: %s" % (label, ", ".join(lost)))
    if requested.get("unknown_family"):
        notes.append("%s: %d scene(s) record no drawn family (artifact predates "
                     "stats.auto_pick and the replay is off)" % (label, requested["unknown_family"]))
    if (requested.get("basis") or {}).get("replayed"):
        notes.append("%s: %d drawn famil(ies) re-drawn with the current generator "
                     "(layout_report.json records no stats.auto_pick)"
                     % (label, requested["basis"]["replayed"]))
    return notes


def bias_block(generation_records: Sequence[dict], generated: dict,
               product_records: Sequence[dict], success_records: Sequence[dict],
               success: dict) -> dict:
    """How far the built and the delivered mix moved away from the request.

    The two comparisons answer different questions: *generation* (drawn family
    vs the specs the generator produced) isolates the fallback inside the
    generator, *product* (drawn family vs the scenes the pipeline delivered)
    adds the failure filtering on top of it.  Each row carries the drawn count,
    the delivered count, the share delta and the **survival** of the family --
    of the seeds that asked for it, how many were delivered as it.
    """
    def compare(records: Sequence[dict], actual: Sequence[dict], block: dict,
                label: str, level: str) -> dict:
        requested = requested_stats(records)
        req, act = requested.get("share") or {}, block.get("family_share") or {}
        counts_req = requested.get("families") or {}
        counts_act = block.get("families") or {}
        survival = _survival(records, actual)
        rows = []
        for family in sorted(set(req) | set(act) | set(counts_req) | set(counts_act)
                             | set(survival)):
            kept = survival.get(family) or {"asked": 0, "kept": 0}
            rows.append({"family": family,
                         "requested": counts_req.get(family, 0),
                         "built": counts_act.get(family, 0),
                         "requested_share": round(req.get(family, 0.0), 4),
                         "actual_share": round(act.get(family, 0.0), 4),
                         "share_delta": round(act.get(family, 0.0) - req.get(family, 0.0), 4),
                         "asked": kept["asked"], "kept": kept["kept"],
                         "survival": round(kept["kept"] / kept["asked"], 4)
                         if kept["asked"] else None})
        return {"level": level,
                "tv_distance": _tv_distance(req, act),
                "families": rows,
                "notes": _bias_notes(requested, block, label)}

    return {"generation": compare(generation_records, generation_records, generated,
                                  "generated", "layout_ok"),
            "product": compare(product_records, success_records, success,
                               "success", "bundle_ok")}


def success_payload(blocks: Sequence[dict], scenes: bool = False) -> dict:
    """The whole success report: request, generated mix, delivered mix, the gap."""
    product = [block for block in blocks if block["role"] == "product"] or list(blocks)
    product_records = _merge_by_seed(product)
    requested = _merge_by_seed(blocks)
    generated = [record for record in requested
                 if (record.get("ok") or {}).get("layout_ok") and record.get("metrics")]
    success = [record for record in product_records if (record.get("ok") or {}).get("bundle_ok")]

    funnel: Dict[str, Optional[int]] = {"seeds": len(product_records)}
    levels = list(SUCCESS_LEVELS)
    if any((record.get("ok") or {}).get("review_ok") is not None for record in product_records):
        levels.append("review_ok")
    for level in levels:
        values = [(record.get("ok") or {}).get(level) for record in product_records]
        # a level the batch never asked for stays null: a layout-only batch must
        # not read as "0 scenes rendered"
        funnel[level] = None if all(value is None for value in values) \
            else sum(1 for value in values if value is True)

    generated_block = _set_block(generated)
    success_block = _set_block(success)
    requested_generation = requested_stats(generated or requested)
    requested_product = requested_stats(product_records)

    payload = {
        "kind": "panoworld_success_diversity",
        "schema_version": 1,
        "created_utc": _utc_now(),
        "roots": [{"root": block["root"], "source": block["source"],
                   "batch_id": block["batch_id"], "role": block["role"],
                   "count": block["count"], "levels": block["levels"],
                   "render_requested": block["render_requested"],
                   "review_requested": block["review_requested"],
                   "request": block["request"], "shape": block["shape"],
                   "rooms": block["rooms"]} for block in blocks],
        "product_roots": [block["root"] for block in product],
        "levels": funnel,
        "requested": {"generation": requested_generation,
                      "product": requested_product},
        "generated": generated_block,
        "success": success_block,
        "filtered": _filtered_block(product_records, success, requested),
        "bias": bias_block(generated, generated_block, product_records, success, success_block),
        "note": "generated = specs that passed the layout stage (every root); "
                "success = bundle_ok products of the roots marked 'product'. "
                "'requested' is what the seed asked for: stats.auto_pick when the "
                "artifact records it, a re-draw with the current generator for "
                "legacy reports (basis 'replayed'), else the declared shape.",
    }
    all_seeds = {record["seed"] for record in requested}
    product_seeds = {record["seed"] for record in product_records}
    if product_seeds and product_seeds < all_seeds:
        payload["sample"] = {
            "product_seeds": len(product_seeds),
            "generated_seeds": len(all_seeds),
            "covered_share": round(len(product_seeds) / len(all_seeds), 4),
            "note": "the product batch covers %d of the %d requested seeds, so its mix is the "
                    "sample's, not the sweep's" % (len(product_seeds), len(all_seeds))}
        payload["bias"]["product"]["notes"].insert(0, payload["sample"]["note"])
    if scenes:
        payload["scenes"] = {"generated": [_scene_payload(record) for record in generated],
                             "success": [_scene_payload(record) for record in success],
                             "filtered": [_scene_payload(record) for record in product_records
                                          if record["seed"] not in {row["seed"]
                                                                    for row in success}]}
    return payload


def _scene_payload(record: dict) -> dict:
    out = {key: value for key, value in record.items() if key not in ("spec", "metrics")}
    out["metrics"] = record.get("metrics")
    return out


def _print_success(payload: dict) -> None:
    roots = ", ".join("%s (%s)" % (os.path.basename(block["root"]), block["role"])
                      for block in payload["roots"])
    print("roots          : %s" % roots)
    print("products       : %d of %d product seeds, %d generated (product roots: %s)"
          % (payload["filtered"]["kept"], payload["filtered"]["requested"],
             payload["filtered"]["generated_requested"],
             ", ".join(os.path.basename(path) for path in payload["product_roots"])))
    print("levels         : %s" % payload["levels"])
    print("filtered       : %s" % (payload["filtered"]["reasons"] or "nothing dropped"))
    for name in ("generation", "product"):
        requested = payload["requested"][name]
        print("requested (%s) : %s (basis %s)" % (name, requested["share"], requested["basis"]))
    print("generated fam. : %s" % payload["generated"].get("family_share"))
    print("success fam.   : %s" % payload["success"].get("family_share"))
    print("floors         : %s" % payload["success"].get("floors"))
    print("shape          : %s" % payload["success"].get("shape"))
    print("shape bins     : %s" % payload["success"].get("shape_bins"))
    print("doors          : %s" % payload["success"].get("door_graph"))
    fallback = payload["success"].get("fallback") or {}
    switch = fallback.get("family_switch") or {}
    auto = fallback.get("auto_fallback") or {}
    print("fallback       : first attempt %s, retried %s, auto fallback %s (of %s known), "
          "family switch %s (recorded draws)"
          % (fallback.get("first_attempt_rate"), fallback.get("retry_rate"),
             "not recorded" if auto.get("rate") is None else auto.get("rate"), auto.get("known"),
             "not recorded" if switch.get("rate") is None else switch.get("rate")))
    for basis in ("replayed", "declared"):
        bucket = (switch.get("by_basis") or {}).get(basis)
        if bucket:
            print("  - %s draws: %s of %s differ from the built family (not a recorded fallback)"
                  % (basis, bucket["switched"], bucket["known"]))
    reuse = payload["success"].get("furniture_reuse") or {}
    print("furniture      : %s placements, %s distinct, reuse %s, cross-scene %s"
          % (reuse.get("placements"), reuse.get("distinct_assets"), reuse.get("reuse_rate"),
             reuse.get("cross_scene_reuse_rate")))
    views = payload["success"].get("view_coverage") or {}
    print("camera views   : %s cameras (%s/scene), rooms without camera %s, coverage %s"
          % (views.get("cameras"), (views.get("cameras_per_scene") or {}).get("mean"),
             views.get("rooms_without_camera"),
             views.get("scene_visible_area_fraction")))
    quality = payload["success"].get("quality") or {}
    rendered = payload["levels"].get("render_ok") is not None
    print("quality        : review %s (score mean %s), validation %s, degraded %s"
          % ("%s/%s passed" % (quality.get("review_passed"), quality.get("reviewed"))
             if quality.get("reviewed") else "not run",
             (quality.get("review_score") or {}).get("mean"),
             "%s at resolution" % quality.get("validation_passed_at_resolution")
             if rendered else "not run",
             quality.get("degraded")))
    for name in ("generation", "product"):
        bias = payload["bias"][name]
        print("bias %-10s : tv(request, actual) %s" % (name, bias["tv_distance"]))
        for note in bias["notes"][:4]:
            print("  - %s" % note)


# --------------------------------------------------------------------------- #
# benchmark: layout sweep -> diversity -> stratified sample -> render + review
# --------------------------------------------------------------------------- #
def _run_batch(start: int, count: int, root: str, env_overrides: Dict[str, str],
               seeds: Optional[Sequence[int]] = None) -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("PANOWORLD_")}
    env.update(env_overrides)
    if seeds is not None:
        env["PANOWORLD_BATCH_SEEDS"] = ",".join(str(seed) for seed in seeds)
    started = datetime.datetime.now(datetime.timezone.utc)
    os.makedirs(root, exist_ok=True)
    log_path = os.path.join(root, "batch_run.log")
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(["bash", DEFAULT_BATCH, str(start), str(count), root],
                                env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    summary_path = os.path.join(root, "batch_summary.json")
    summary = None
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    return {
        "exit_code": result.returncode,
        "seconds": round((datetime.datetime.now(datetime.timezone.utc) - started).total_seconds(), 2),
        "summary": summary,
        "summary_path": summary_path if summary is not None else None,
        "log": log_path,
    }


def _blender_running() -> List[str]:
    try:
        result = subprocess.run(["pgrep", "-a", "blender"], capture_output=True, text=True)
    except Exception:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def choose_sample(metrics: Sequence[dict], size: int) -> dict:
    """Stratified sample: one per family, then new structures, then seed order."""
    by_family: Dict[str, List[dict]] = {}
    for entry in metrics:
        by_family.setdefault(str(entry.get("family")), []).append(entry)
    for entries in by_family.values():
        entries.sort(key=lambda entry: (entry.get("seed") if entry.get("seed") is not None else -1))

    chosen: List[dict] = []
    reasons: Dict[int, str] = {}
    for family in sorted(by_family):
        if len(chosen) >= size:
            break
        entry = by_family[family][0]
        chosen.append(entry)
        reasons[entry["seed"]] = f"first of family '{family}'"

    seen_structures = {entry["structure_key"] for entry in chosen}
    for entry in sorted(metrics, key=lambda item: item.get("seed") if item.get("seed") is not None else -1):
        if len(chosen) >= size:
            break
        if entry in chosen or entry["structure_key"] in seen_structures:
            continue
        seen_structures.add(entry["structure_key"])
        chosen.append(entry)
        reasons[entry["seed"]] = "new topology structure"
    for entry in sorted(metrics, key=lambda item: item.get("seed") if item.get("seed") is not None else -1):
        if len(chosen) >= size:
            break
        if entry in chosen:
            continue
        chosen.append(entry)
        reasons[entry["seed"]] = "fill"
    chosen.sort(key=lambda entry: entry["seed"])
    return {
        "seeds": [entry["seed"] for entry in chosen],
        "families": {str(entry["seed"]): entry.get("family") for entry in chosen},
        "reasons": {str(seed): reason for seed, reason in reasons.items()},
        "per_family": {family: sum(1 for entry in chosen if str(entry.get("family")) == family)
                       for family in sorted({str(entry.get("family")) for entry in metrics})},
    }


def _per_scene_costs(root: str, seeds: Sequence[int]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for seed in seeds:
        scene = os.path.join(root, "scene_%04d" % seed)
        summary_path = os.path.join(scene, "run_summary.json")
        if not os.path.isfile(summary_path):
            continue
        try:
            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
        except Exception:
            continue
        review = summary.get("review") or {}
        out[str(seed)] = {
            "seconds": summary.get("elapsed_seconds"),
            "status": "bundle_ok" if (summary.get("ok") or {}).get("bundle_ok") else "failed",
            "render": (summary.get("render") or {}).get("status"),
            "review": review.get("status"),
            "review_pass": review.get("pass"),
            "review_issues": review.get("issues"),
            "review_failed_checks": review.get("failed_checks"),
            "repair": review.get("repair"),
            "resolution": (summary.get("render") or {}).get("resolution"),
        }
    return out


def benchmark(args) -> int:
    root = os.path.abspath(args.root)
    seeds = parse_seeds(args.seeds)
    if not seeds:
        print("benchmark: --seeds is empty", file=sys.stderr)
        return 2
    os.makedirs(root, exist_ok=True)
    started = _utc_now()
    report: dict = {
        "kind": "panoworld_benchmark",
        "schema_version": 1,
        "started_at": started,
        "root": root,
        "seeds": {"range": [seeds[0], seeds[-1]], "count": len(seeds)},
        "config": {"sample": args.sample, "shape": args.shape, "resolution": args.resolution,
                   "samples": args.samples, "min_free_mb": args.min_free_mb,
                   "review": True, "repair": bool(args.repair),
                   "layout_batch": os.path.join(root, "layout"),
                   "sample_batch": os.path.join(root, "sample")},
    }

    # -- 1: layout over every seed ------------------------------------------ #
    print(f"[1/4] layout sweep: {len(seeds)} seeds -> {os.path.join(root, 'layout')}")
    layout_env = {"PANOWORLD_BATCH_RESUME": "1" if args.resume else "0",
                  "PANOWORLD_LAYOUT_SHAPE": args.shape}
    if args.rooms:
        layout_env["PANOWORLD_ROOMS"] = str(args.rooms)
    layout = _run_batch(seeds[0], len(seeds), report["config"]["layout_batch"], layout_env,
                        seeds=seeds)
    layout_summary = layout["summary"] or {}
    report["layout"] = {
        "seconds": layout["seconds"],
        "exit_code": layout["exit_code"],
        "levels": layout_summary.get("levels"),
        "passed": layout_summary.get("passed"),
        "failed": layout_summary.get("failed"),
        "failure_reasons": layout_summary.get("failure_reasons"),
        "batch_summary": layout["summary_path"],
        "log": layout["log"],
    }
    ok_seeds = [row["seed"] for row in layout_summary.get("scenes", [])
                if (row.get("ok") or {}).get("layout_ok")]
    layout_seconds = [entry["seconds"] for entry in
                      _per_scene_costs(report["config"]["layout_batch"], seeds).values()
                      if entry.get("seconds")]
    if layout_seconds:
        report["layout"]["per_scene_seconds"] = {
            "mean": round(statistics.fmean(layout_seconds), 2),
            "median": round(statistics.median(layout_seconds), 2),
            "max": max(layout_seconds)}
    print(f"      layout_ok {layout_summary.get('levels', {}).get('layout_ok')}/{len(seeds)}"
          f" in {layout['seconds']:.1f} s"
          f" (~{report['layout'].get('per_scene_seconds', {}).get('mean')} s/scene)")

    # -- 2: diversity ------------------------------------------------------- #
    print("[2/4] diversity metrics")
    specs = load_specs([os.path.join(report["config"]["layout_batch"], "scene_*", "spec.json")])
    metrics = []
    for entry in specs:
        seed = None
        if os.path.basename(os.path.dirname(entry["path"])).startswith("scene_"):
            try:
                seed = int(os.path.basename(os.path.dirname(entry["path"]))[len("scene_"):])
            except ValueError:
                seed = None
        metrics.append(scene_metrics(entry["spec"], seed=seed,
                                     layout_report=entry["layout_report"], path=entry["path"]))
    summary = diversity_summary(metrics)
    report["diversity"] = dict(summary, scenes=metrics)
    with open(os.path.join(root, "diversity.json"), "w", encoding="utf-8") as handle:
        json.dump({"kind": "panoworld_diversity", "created_utc": _utc_now(),
                   "count": len(metrics), "summary": summary, "scenes": metrics},
                  handle, indent=2)
    print(f"      {summary.get('count')} structures: {summary.get('structures', {}).get('unique')} "
          f"unique (duplicate rate {summary.get('structures', {}).get('duplicate_rate')}), "
          f"plan duplicates {summary.get('plans', {}).get('duplicate_rate')}")

    # -- 3: the stratified sample ------------------------------------------ #
    sample = choose_sample(metrics, max(1, args.sample))
    report["sample"] = sample
    print(f"[3/4] sample: {sample['seeds']} (per family {sample['per_family']})")

    # -- 4: render + review the sample ------------------------------------- #
    if args.no_render:
        report["render"] = {"status": "skipped_by_request"}
    else:
        running = _blender_running()
        if running and not args.force:
            print("benchmark: another blender process is running:\n  " + "\n  ".join(running[:5]),
                  file=sys.stderr)
            print("benchmark: refusing to start renders (use --force to override)", file=sys.stderr)
            report["render"] = {"status": "blocked_by_concurrent_blender", "processes": running[:5]}
        else:
            print(f"[4/4] render + review {len(sample['seeds'])} scenes -> "
                  f"{report['config']['sample_batch']}")
            render_env = {"PANOWORLD_BATCH_RENDER": "1", "PANOWORLD_REVIEW": "1",
                          "PANOWORLD_BATCH_RESUME": "1" if args.resume else "0",
                          "PANOWORLD_RES": args.resolution, "PANOWORLD_SAMPLES": str(args.samples),
                          "PANOWORLD_MIN_FREE_MB": str(args.min_free_mb),
                          "PANOWORLD_LAYOUT_SHAPE": args.shape}
            if args.repair:
                render_env["PANOWORLD_REVIEW_REPAIR"] = "1"
            if args.rooms:
                render_env["PANOWORLD_ROOMS"] = str(args.rooms)
            rendered = _run_batch(seeds[0], len(sample["seeds"]), report["config"]["sample_batch"],
                                  render_env, seeds=sample["seeds"])
            rendered_summary = rendered["summary"] or {}
            costs = _per_scene_costs(report["config"]["sample_batch"], sample["seeds"])
            seconds = [entry["seconds"] for entry in costs.values() if entry.get("seconds")]
            report["render"] = {
                "seconds": rendered["seconds"],
                "exit_code": rendered["exit_code"],
                "levels": rendered_summary.get("levels"),
                "passed": rendered_summary.get("passed"),
                "pending_retry": rendered_summary.get("pending_retry"),
                "failed": rendered_summary.get("failed"),
                "failure_reasons": rendered_summary.get("failure_reasons"),
                "failed_checks": rendered_summary.get("failed_checks"),
                "review_issues": rendered_summary.get("review_issues"),
                "per_scene": costs,
                "per_scene_seconds": {"mean": round(statistics.fmean(seconds), 1) if seconds else None,
                                      "median": round(statistics.median(seconds), 1) if seconds else None,
                                      "max": max(seconds) if seconds else None},
                "batch_summary": rendered["summary_path"],
                "log": rendered["log"],
            }
            print(f"      bundle_ok {rendered_summary.get('passed')}/{len(sample['seeds'])}, "
                  f"review_ok {(rendered_summary.get('levels') or {}).get('review_ok')}"
                  f"/{len(sample['seeds'])} in {rendered['seconds']:.1f} s"
                  f" (~{report['render']['per_scene_seconds']['mean']} s/scene)")

    report["finished_at"] = _utc_now()
    report_path = os.path.join(root, "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"benchmark report: {report_path}")
    layout_ok = report["layout"].get("passed") == len(seeds)
    render = report.get("render") or {}
    render_ok = render.get("status") is None and render.get("passed") == len(sample["seeds"])
    return 0 if (layout_ok and (args.no_render or render_ok)) else 1


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def _write(path: Optional[str], payload: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def success_command(args) -> int:
    """Read every batch root, compare the request with the delivered products."""
    try:
        blocks = _set_role([read_root(root, replay_draw=not args.no_replay_draw)
                            for root in args.roots], args.product_root)
    except ValueError as error:
        print("success: %s" % error, file=sys.stderr)
        return 2
    payload = success_payload(blocks, scenes=args.scenes)
    if args.out:
        _write(args.out, payload)
        print("success report: %s" % args.out)
    _print_success(payload)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Structural diversity of generated houses")
    sub = parser.add_subparsers(dest="command", required=True)

    specs_cmd = sub.add_parser("specs", help="measure existing spec files / batch roots")
    specs_cmd.add_argument("paths", nargs="+")
    specs_cmd.add_argument("--out", default=None)
    specs_cmd.add_argument("--scenes", action="store_true", help="keep the per-scene metrics")

    seeds_cmd = sub.add_parser("seeds", help="generate specs in process and measure them")
    seeds_cmd.add_argument("--seeds", default="0-99")
    seeds_cmd.add_argument("--shape", default="auto")
    seeds_cmd.add_argument("--rooms", type=int, default=None)
    seeds_cmd.add_argument("--out", default=None)
    seeds_cmd.add_argument("--scenes", action="store_true")
    seeds_cmd.add_argument("--hostpy", default=DEFAULT_HOSTPY,
                           help="interpreter that generates the specs")

    success_cmd = sub.add_parser("success", help="final products: family mix, fallback and bias")
    success_cmd.add_argument("--root", action="append", required=True, dest="roots",
                             help="batch root with batch_summary.json + scene_*/ (repeatable)")
    success_cmd.add_argument("--product-root", action="append", default=None,
                             help="root whose bundle_ok scenes are the final products "
                                  "(default: every root that requested a render, else every root)")
    success_cmd.add_argument("--no-replay-draw", action="store_true",
                             help="do not re-draw the family of artifacts that record none")
    success_cmd.add_argument("--out", default=None)
    success_cmd.add_argument("--scenes", action="store_true", help="keep the per-scene records")

    bench_cmd = sub.add_parser("benchmark", help="layout sweep + stratified render/review sample")
    bench_cmd.add_argument("--seeds", default="0-99")
    bench_cmd.add_argument("--sample", type=int, default=10, help="scenes to render (8-12)")
    bench_cmd.add_argument("--root", default=os.path.join(REPO_ROOT, "outputs", "benchmark100"))
    bench_cmd.add_argument("--shape", default="auto")
    bench_cmd.add_argument("--rooms", type=int, default=None)
    bench_cmd.add_argument("--resolution", default="1024x512")
    bench_cmd.add_argument("--samples", type=int, default=24)
    bench_cmd.add_argument("--min-free-mb", type=int, default=2200,
                           help="layout_gen_run.sh render gate (cgroup headroom)")
    bench_cmd.add_argument("--repair", action="store_true", help="PANOWORLD_REVIEW_REPAIR=1")
    bench_cmd.add_argument("--resume", action="store_true", help="reuse completed scenes")
    bench_cmd.add_argument("--no-render", action="store_true")
    bench_cmd.add_argument("--force", action="store_true",
                           help="render even when another blender process is running")

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        return benchmark(args)

    if args.command == "success":
        return success_command(args)

    if args.command == "specs":
        entries = load_specs(args.paths)
        metrics = [scene_metrics(entry["spec"], layout_report=entry["layout_report"],
                                 path=entry["path"]) for entry in entries]
    else:
        seeds = parse_seeds(args.seeds)
        generated = generate_specs(seeds, args.hostpy, args.shape, args.rooms)
        metrics = [scene_metrics(spec, seed=seed, layout_report=report)
                   for seed, spec, report in generated]

    summary = diversity_summary(metrics)
    payload = {"kind": "panoworld_diversity", "created_utc": _utc_now(),
               "count": len(metrics), "summary": summary}
    if args.scenes:
        payload["scenes"] = metrics
    if args.out:
        _write(args.out, payload)
    _print_summary(summary)
    return 0


def generate_specs(seeds: Sequence[int], hostpy: str, shape: str,
                   rooms: Optional[int] = None) -> List[Tuple[int, dict, Optional[dict]]]:
    """Write each seed's spec through the generator CLI and read it back."""
    import tempfile

    out: List[Tuple[int, dict, Optional[dict]]] = []
    tmpdir = tempfile.mkdtemp(prefix="diversity_")
    for seed in seeds:
        path = os.path.join(tmpdir, f"seed_{seed}.json")
        report_path = os.path.join(tmpdir, f"seed_{seed}_report.json")
        argv = [hostpy, os.path.join(REPO_ROOT, "blender_agent", "layout_gen.py"),
                "--seed", str(seed), "--out", path, "--shape", shape, "--quiet",
                "--report", report_path]
        if rooms:
            argv += ["--rooms", str(rooms)]
        result = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT)
        if result.returncode != 0 or not os.path.isfile(path):
            print(f"diversity: seed {seed} failed: {result.stderr.strip()[-200:]}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as handle:
            spec = json.load(handle)
        report = None
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
        out.append((seed, spec, report))
    return out


def _print_summary(summary: dict) -> None:
    print(f"scenes            : {summary.get('count')}")
    print(f"families          : {summary.get('families')}")
    print(f"rooms (histogram) : {summary.get('rooms', {}).get('histogram')}")
    print(f"floors (storeys)  : {summary.get('floors')}")
    print(f"connections       : {summary.get('connections')}")
    print(f"cycles            : {summary.get('cycles')}")
    print(f"function coverage : {summary.get('function_coverage')}")
    print(f"shape             : {summary.get('shape')}")
    print(f"furniture         : {summary.get('furniture')}")
    print(f"cameras           : {summary.get('cameras')}")
    structures = summary.get("structures", {})
    print(f"topologies        : {structures.get('unique')} unique of {summary.get('count')} "
          f"(duplicate rate {structures.get('duplicate_rate')})")
    for group in structures.get("groups", [])[:5]:
        print(f"  repeat x{group['count']:>2}  {group['hash']}  seeds {group['seeds'][:8]}  "
              f"{group['note']}")
    plans = summary.get("plans", {})
    print(f"plans             : {plans.get('unique')} unique (duplicate rate {plans.get('duplicate_rate')})")


if __name__ == "__main__":
    sys.exit(main())
