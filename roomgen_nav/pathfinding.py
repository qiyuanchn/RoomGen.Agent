"""Pure geometry, composite path planning and path validation.

This module deliberately has no Habitat-Sim import.  ``navmesh.py`` supplies
the per-storey ``PathFinder`` adapters; tests can drive the planner with tiny
fake floors.  The planner models a building as:

* one navigable surface per storey (its baked NavMesh);
* one *link* per stair / ramp walk track, with two snapped gate points and the
  ordered 3D track between them.

A composite route is the shortest path in a small graph whose nodes are the
start, the goal and every link gate.  Floor edges are geodesic NavMesh paths,
link edges are the walk track; Dijkstra therefore decides both which rooms to
cross and which stair / ramp to use.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from blender_agent import spec as spec_mod

Point = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #
def as_point(value: Sequence[float]) -> Point:
    return (float(value[0]), float(value[1]), float(value[2]))


def point_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def polyline_length(points: Sequence[Sequence[float]]) -> float:
    return sum(point_distance(points[i], points[i + 1]) for i in range(len(points) - 1))


def heading_deg(p0: Sequence[float], p1: Sequence[float]) -> float:
    """Compass-style heading in the XZ plane: 0 = +X, 90 = -Z (Y-up frame)."""
    dx = float(p1[0]) - float(p0[0])
    dz = float(p1[2]) - float(p0[2])
    if abs(dx) < 1e-12 and abs(dz) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(-dz, dx))


def wrap_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def angle_delta_deg(target: float, current: float) -> float:
    return wrap_angle_deg(float(target) - float(current))


def forward_vector(yaw_deg: float) -> Tuple[float, float]:
    """XZ forward vector for a heading; matches ``heading_deg``."""
    angle = math.radians(float(yaw_deg))
    return (math.cos(angle), -math.sin(angle))


def densify_polyline(points: Sequence[Sequence[float]],
                     max_spacing: float = 0.15) -> List[Point]:
    """Insert points so no 3D segment is longer than ``max_spacing``."""
    if not points:
        return []
    dense: List[Point] = [as_point(points[0])]
    for index in range(1, len(points)):
        start = dense[-1]
        end = as_point(points[index])
        length = point_distance(start, end)
        count = max(1, int(math.ceil(length / max_spacing)))
        for step in range(1, count + 1):
            t = step / count
            dense.append((start[0] + (end[0] - start[0]) * t,
                          start[1] + (end[1] - start[1]) * t,
                          start[2] + (end[2] - start[2]) * t))
    return dense


def point_segment_distance_2d(point: Sequence[float], a: Sequence[float],
                              b: Sequence[float]) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 < 1e-18:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_polygon_boundary(point: Sequence[float],
                                 polygon: Sequence[Sequence[float]]) -> float:
    return min(point_segment_distance_2d(point, polygon[i], polygon[(i + 1) % len(polygon)])
               for i in range(len(polygon)))


def sample_point_in_polygon(rng, polygon: Sequence[Sequence[float]],
                            margin: float = 0.35,
                            max_tries: int = 500) -> Optional[Tuple[float, float]]:
    """Deterministic rejection sample inside ``polygon`` at least ``margin`` from its edges."""
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    for _ in range(max_tries):
        x = rng.uniform(x0, x1)
        y = rng.uniform(y0, y1)
        if spec_mod.point_in_polygon((x, y), polygon) and \
                distance_to_polygon_boundary((x, y), polygon) >= margin:
            return (x, y)
    return None


def segment_crosses_polygon_2d(p0: Sequence[float], p1: Sequence[float],
                               polygon: Sequence[Sequence[float]],
                               step: float = 0.05) -> bool:
    """True when any sample on the XZ segment lies inside ``polygon``."""
    dx = float(p1[0]) - float(p0[0])
    dz = float(p1[2]) - float(p0[2])
    length = math.hypot(dx, dz)
    count = max(1, int(math.ceil(length / step)))
    for index in range(count + 1):
        t = index / count
        point = (float(p0[0]) + dx * t, float(p0[2]) + dz * t)
        if spec_mod.point_in_polygon(point, polygon):
            return True
    return False


# --------------------------------------------------------------------------- #
# path records
# --------------------------------------------------------------------------- #
@dataclass
class FloorPath:
    """A geodesic path on one storey's NavMesh (Y-up habitat coordinates)."""

    points: List[Point]
    distance: float


@dataclass
class LinkEndpoint:
    floor_id: str
    point: Point
    label: str = ""


@dataclass
class NavLink:
    """A stair / ramp walk track joining two floor gates.

    The track is stored in the traversal direction ``from`` -> ``to``, including
    both snapped gate points.  ``valid`` is False when the profile cannot use it
    (riser too high, ramp too steep, gate off the NavMesh); ``reason`` says why,
    and the planner never uses an invalid link.
    """

    link_id: str
    kind: str
    from_room: str
    to_room: str
    from_floor: str
    to_floor: str
    from_endpoint: LinkEndpoint
    to_endpoint: LinkEndpoint
    track: List[Point]
    valid: bool = True
    reason: str = ""
    connector_index: int = -1

    @property
    def distance(self) -> float:
        return polyline_length(self.track)

    @property
    def is_cross_floor(self) -> bool:
        return self.from_floor != self.to_floor

    def endpoints_for(self, floor_id: str) -> List[LinkEndpoint]:
        out = []
        if self.from_floor == floor_id:
            out.append(self.from_endpoint)
        if self.to_floor == floor_id:
            out.append(self.to_endpoint)
        return out

    def to_dict(self) -> dict:
        return {
            "link_id": self.link_id, "kind": self.kind,
            "from_room": self.from_room, "to_room": self.to_room,
            "from_floor": self.from_floor, "to_floor": self.to_floor,
            "from_point": list(self.from_endpoint.point),
            "to_point": list(self.to_endpoint.point),
            "distance": round(self.distance, 4),
            "valid": self.valid, "reason": self.reason,
            "track": [list(p) for p in self.track],
        }


@dataclass
class PathSegment:
    """One edge of a composite route: a floor geodesic or a link track."""

    kind: str  # "floor" | "link"
    floor_id: str
    points: List[Point]
    distance: float
    link_id: Optional[str] = None
    direction: Optional[str] = None


@dataclass
class CompositePath:
    points: List[Point]
    distance: float
    segments: List[PathSegment] = field(default_factory=list)
    links_used: List[str] = field(default_factory=list)

    @property
    def crosses_floors(self) -> bool:
        floors = {segment.floor_id for segment in self.segments if segment.kind == "floor"}
        return len(floors) > 1

    def to_dict(self) -> dict:
        return {
            "distance": round(self.distance, 4),
            "points": [list(p) for p in self.points],
            "links_used": list(self.links_used),
            "crosses_floors": self.crosses_floors,
            "segments": [{"kind": s.kind, "floor_id": s.floor_id,
                          "distance": round(s.distance, 4), "link_id": s.link_id,
                          "direction": s.direction} for s in self.segments],
        }


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
def _snap(floor_nav, point: Sequence[float], tolerance: float) -> Optional[Point]:
    snapped = floor_nav.snap_point(point)
    if snapped is None:
        return None
    snapped = as_point(snapped)
    if not all(math.isfinite(value) for value in snapped):
        return None
    if point_distance(snapped, point) > tolerance:
        return None
    return snapped


def _dedupe_points(points: Sequence[Point], eps: float = 1e-6) -> List[Point]:
    out: List[Point] = []
    for point in points:
        if not out or point_distance(out[-1], point) > eps:
            out.append(as_point(point))
    return out


def plan_composite_path(floor_navs: Dict[str, object],
                        links: Sequence[NavLink],
                        start_floor: str,
                        start_point: Sequence[float],
                        goal_floor: str,
                        goal_point: Sequence[float],
                        snap_tolerance: float = 0.75,
                        include_start_goal_snap: bool = True) -> Optional[CompositePath]:
    """Shortest route through floor NavMeshes and stair / ramp links.

    ``floor_navs`` maps floor id to an object with ``snap_point(point)`` and
    ``find_path(start, end) -> FloorPath | None``.  Returns None when the start
    or goal cannot be snapped, or when no route exists.
    """
    if start_floor not in floor_navs or goal_floor not in floor_navs:
        raise KeyError(f"missing floor NavMesh: start={start_floor} goal={goal_floor}")
    start = _snap(floor_navs[start_floor], start_point, snap_tolerance) \
        if include_start_goal_snap else as_point(start_point)
    goal = _snap(floor_navs[goal_floor], goal_point, snap_tolerance) \
        if include_start_goal_snap else as_point(goal_point)
    if start is None or goal is None:
        return None

    # Nodes: start, goal, then both gates of every valid link.
    node_floor: List[str] = [start_floor, goal_floor]
    node_point: List[Point] = [start, goal]
    node_label: List[str] = ["start", "goal"]
    link_nodes: List[Tuple[int, int]] = []
    for link in links:
        if not link.valid:
            link_nodes.append((-1, -1))
            continue
        index_from = len(node_floor)
        node_floor.append(link.from_floor)
        node_point.append(as_point(link.from_endpoint.point))
        node_label.append(f"{link.link_id}:from")
        index_to = len(node_floor)
        node_floor.append(link.to_floor)
        node_point.append(as_point(link.to_endpoint.point))
        node_label.append(f"{link.link_id}:to")
        link_nodes.append((index_from, index_to))

    count = len(node_floor)
    adjacency: List[List[Tuple[int, float, str, object]]] = [[] for _ in range(count)]
    path_cache: Dict[Tuple[int, int], Optional[FloorPath]] = {}

    def floor_edge(i: int, j: int) -> Optional[FloorPath]:
        key = (i, j)
        if key in path_cache:
            return path_cache[key]
        if node_floor[i] != node_floor[j]:
            path_cache[key] = None
            return None
        result = floor_navs[node_floor[i]].find_path(node_point[i], node_point[j])
        if isinstance(result, FloorPath):
            path = result
        elif result is None:
            path = None
        else:  # adapter returned (points, distance)
            points, distance = result
            path = FloorPath([as_point(p) for p in points], float(distance))
        path_cache[key] = path
        return path

    for i in range(count):
        for j in range(i + 1, count):
            if node_floor[i] != node_floor[j]:
                continue
            path = floor_edge(i, j)
            if path is None or path.distance <= 1e-9:
                continue
            adjacency[i].append((j, path.distance, "floor", path))
            adjacency[j].append((i, path.distance, "floor", path))

    for (index_from, index_to), link in zip(link_nodes, links):
        if index_from < 0 or not link.valid:
            continue
        adjacency[index_from].append((index_to, link.distance, "link", (link, "forward")))
        adjacency[index_to].append((index_from, link.distance, "link", (link, "reverse")))

    # Dijkstra with deterministic tie-breaking by node index.
    inf = float("inf")
    dist = [inf] * count
    previous: List[Optional[Tuple[int, float, str, object]]] = [None] * count
    dist[0] = 0.0
    heap = [(0.0, 0)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist[node] + 1e-12:
            continue
        if node == 1:
            break
        for target, weight, kind, payload in adjacency[node]:
            new_cost = cost + weight
            if new_cost + 1e-12 < dist[target]:
                dist[target] = new_cost
                previous[target] = (node, weight, kind, payload)
                heapq.heappush(heap, (new_cost, target))
    if not math.isfinite(dist[1]):
        return None

    edges = []
    cursor = 1
    while cursor != 0:
        step = previous[cursor]
        if step is None:
            return None
        edges.append((step[0], cursor, step[1], step[2], step[3]))
        cursor = step[0]
    edges.reverse()

    points: List[Point] = []
    segments: List[PathSegment] = []
    links_used: List[str] = []
    for source, target, weight, kind, payload in edges:
        if kind == "floor":
            floor_path: FloorPath = payload
            part = list(floor_path.points)
            if source > target:
                part.reverse()
            part = _dedupe_points(part)
            if not part:
                continue
            segments.append(PathSegment("floor", node_floor[source], part,
                                        floor_path.distance))
            points.extend(part if not points else part[1:])
        else:
            link, direction = payload
            track = list(link.track)
            if direction == "reverse":
                track.reverse()
            track = _dedupe_points(track)
            segments.append(PathSegment("link", link.from_floor if direction == "forward"
                                        else link.to_floor, track, link.distance,
                                        link_id=link.link_id, direction=direction))
            links_used.append(link.link_id)
            points.extend(track if not points else track[1:])
    points = _dedupe_points(points)
    if not points:
        return None
    return CompositePath(points, dist[1], segments, links_used)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@dataclass
class PathViolation:
    code: str
    detail: str
    point: Optional[Point] = None

    def to_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail,
                "point": list(self.point) if self.point else None}


def _segment_enters_opening(scene, floor_id: str, a: Sequence[float],
                            b: Sequence[float], step: float = 0.05) -> bool:
    """Sample a floor segment against the storey's slab openings / voids.

    The void polygons are in the spec's Blender XY frame, so the Habitat
    (x, y, z) point maps to ``(x, -z)``.
    """
    dx = float(b[0]) - float(a[0])
    dz = float(b[2]) - float(a[2])
    length = math.hypot(dx, dz)
    count = max(1, int(math.ceil(length / step)))
    for index in range(count + 1):
        t = index / count
        xy = (float(a[0]) + dx * t, -(float(a[2]) + dz * t))
        if scene.in_slab_opening(floor_id, xy):
            return True
    return False


def validate_composite_path(scene: spec_mod.SceneSpec, path: CompositePath,
                            profile, step_tolerance: float = 0.05) -> List[PathViolation]:
    """Catch the failure modes a raw distance cannot see.

    * a floor edge that climbs more than the profile's step height (a split
      level or a badly linked stair gate);
    * a floor edge that crosses a declared void / slab opening on a storey where
      there is no floor (the planner should not have returned it);
    * malformed link tracks.

    Wall / door clearance is already enforced by the Recast bake (agent radius),
    so it is not re-tested here.
    """
    violations: List[PathViolation] = []
    for segment in path.segments:
        if segment.kind == "floor":
            for i in range(len(segment.points) - 1):
                a, b = segment.points[i], segment.points[i + 1]
                rise = abs(float(b[1]) - float(a[1]))
                if rise > profile.max_step + step_tolerance:
                    violations.append(PathViolation(
                        "step_exceeds_max_step",
                        f"{segment.floor_id}: rise {rise:.3f} m > max_step "
                        f"{profile.max_step:.3f} m", a))
                    break
                if scene.floors and _segment_enters_opening(scene, segment.floor_id, a, b):
                    violations.append(PathViolation(
                        "crosses_void",
                        f"{segment.floor_id}: floor path enters a slab opening / void",
                        a))
                    break
        else:
            if len(segment.points) < 2:
                violations.append(PathViolation(
                    "empty_link", f"{segment.link_id}: link track has fewer than 2 points"))
    return violations


def link_profile_problem(link: NavLink, profile) -> Optional[str]:
    """Return why a profile cannot use this link, or None when it can."""
    if len(link.track) < 2:
        return "empty_link_track"
    if link.kind == "stair":
        rises = [abs(link.track[i + 1][1] - link.track[i][1])
                 for i in range(len(link.track) - 1)]
        worst = max(rises) if rises else 0.0
        if not profile.can_climb_riser(worst):
            return f"riser {worst:.3f} m exceeds max_step {profile.max_step:.3f} m"
    elif link.kind == "ramp":
        rise = abs(link.track[-1][1] - link.track[0][1])
        run = math.hypot(link.track[-1][0] - link.track[0][0],
                         link.track[-1][2] - link.track[0][2])
        if not profile.can_traverse_slope(rise, run):
            return (f"slope {rise / run if run > 0 else float('inf'):.3f} exceeds "
                    f"max_slope {profile.max_slope:.3f}")
    else:
        return f"unknown_link_kind:{link.kind}"
    return None


def room_pair_candidates(scene: spec_mod.SceneSpec) -> List[Tuple[str, str]]:
    """Every unordered room pair that a connector joins (for episode targets)."""
    pairs = []
    seen = set()
    for connector in scene.connections():
        key = tuple(sorted((connector.from_room, connector.to_room)))
        if key not in seen:
            seen.add(key)
            pairs.append((connector.from_room, connector.to_room))
    return pairs
