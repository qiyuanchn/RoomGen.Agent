"""Per-storey NavMesh baking and the CompositeNavMesh wrapper.

Why the bake is per storey
--------------------------
Recast (through Habitat-Sim 0.3.3) keeps only one walkable surface when two
storey slabs share the same XY footprint; on a duplex the upper floor simply
disappears from the baked mesh.  Splitting the delivered mesh by storey before
``recompute_navmesh`` gives one NavMesh per floor.  The floors are then joined at
the spec's stair / ramp walk tracks (``spec.walk_tracks()``), which are the only
legal way to change storey.

A profile changes Recast's agent radius / height / climb / slope, so the cache
is keyed by :meth:`AgentProfile.navmesh_fingerprint`; the wheeled profile gets its own
NavMesh and cannot inherit the humanoid bake.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from roomgen_nav import assets
from roomgen_nav.pathfinding import (
    CompositePath,
    FloorPath,
    LinkEndpoint,
    NavLink,
    PathViolation,
    link_profile_problem,
    plan_composite_path,
    point_distance,
    validate_composite_path,
)

try:  # habitat-sim only exists in third_party/nav_env, never in the host env
    import habitat_sim  # type: ignore
    from magnum import Vector3 as _MagnumVector3  # type: ignore
    HABITAT_AVAILABLE = True
    HABITAT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised on the host Python
    habitat_sim = None  # type: ignore
    _MagnumVector3 = None  # type: ignore
    HABITAT_AVAILABLE = False
    HABITAT_IMPORT_ERROR = exc

#: bump when the bake recipe changes so stale caches are discarded
NAV_CACHE_VERSION = "1"


def require_habitat() -> None:
    if not HABITAT_AVAILABLE:
        raise RuntimeError(
            "habitat_sim is not importable in this interpreter; run the nav stack "
            "with third_party/nav_env/bin/python and "
            "LD_LIBRARY_PATH=third_party/nav_sysroot/usr/lib/x86_64-linux-gnu:"
            "third_party/nav_env/lib (import error: "
            f"{HABITAT_IMPORT_ERROR})")


# --------------------------------------------------------------------------- #
# one floor
# --------------------------------------------------------------------------- #
@dataclass
class FloorNavMesh:
    floor_id: str
    elevation: float
    height: float
    navmesh_path: str
    collision_obj: str
    pathfinder: "HabitatFloorPathfinder"
    metadata: dict = field(default_factory=dict)

    @property
    def num_islands(self) -> int:
        return int(self.pathfinder.num_islands)

    @property
    def navigable_area(self) -> float:
        return float(self.pathfinder.navigable_area)

    def snap_point(self, point: Sequence[float]):
        return self.pathfinder.snap_point(point)

    def find_path(self, start: Sequence[float], end: Sequence[float]):
        return self.pathfinder.find_path(start, end)

    def try_step(self, start: Sequence[float], end: Sequence[float]):
        return self.pathfinder.try_step(start, end)

    def to_dict(self) -> dict:
        return {
            "floor_id": self.floor_id,
            "elevation": self.elevation,
            "height": self.height,
            "navmesh": self.navmesh_path,
            "collision_obj": self.collision_obj,
            "num_islands": self.num_islands,
            "navigable_area": round(self.navigable_area, 4),
            **self.metadata,
        }


class HabitatFloorPathfinder:
    """Thin wrapper around a standalone ``habitat_sim.PathFinder``."""

    def __init__(self, navmesh_path: str):
        require_habitat()
        self.navmesh_path = navmesh_path
        self.finder = habitat_sim.PathFinder()
        if not self.finder.load_nav_mesh(navmesh_path):
            raise RuntimeError(f"could not load NavMesh '{navmesh_path}'")

    @property
    def num_islands(self) -> int:
        return int(self.finder.num_islands)

    @property
    def navigable_area(self) -> float:
        return float(self.finder.navigable_area)

    def bounds(self):
        lo, hi = self.finder.get_bounds()
        return ([float(lo[0]), float(lo[1]), float(lo[2])],
                [float(hi[0]), float(hi[1]), float(hi[2])])

    def island_area(self, index: int) -> float:
        try:
            return float(self.finder.island_area(index))
        except Exception:
            return 0.0

    def island_radius(self, index: int) -> float:
        try:
            return float(self.finder.island_radius(index))
        except Exception:
            return 0.0

    def get_island(self, point: Sequence[float]) -> int:
        try:
            return int(self.finder.get_island(_vec(point)))
        except Exception:
            return -1

    def is_navigable(self, point: Sequence[float]) -> bool:
        return bool(self.finder.is_navigable(_vec(point)))

    def snap_point(self, point: Sequence[float]) -> Optional[Tuple[float, float, float]]:
        snapped = self.finder.snap_point(_vec(point))
        values = (float(snapped[0]), float(snapped[1]), float(snapped[2]))
        if not all(math.isfinite(value) for value in values):
            return None
        return values

    def find_path(self, start: Sequence[float], end: Sequence[float]) -> Optional[FloorPath]:
        path = habitat_sim.ShortestPath()
        path.requested_start = _vec(start)
        path.requested_end = _vec(end)
        if not self.finder.find_path(path):
            return None
        points = [(float(p[0]), float(p[1]), float(p[2])) for p in path.points]
        if not points:
            return None
        distance = float(path.geodesic_distance)
        if not math.isfinite(distance):
            return None
        return FloorPath(points, distance)

    def try_step(self, start: Sequence[float], end: Sequence[float]):
        stepped = self.finder.try_step(_vec(start), _vec(end))
        values = (float(stepped[0]), float(stepped[1]), float(stepped[2]))
        if not all(math.isfinite(value) for value in values):
            return None
        return values


def _vec(point: Sequence[float]):
    require_habitat()
    values = (float(point[0]), float(point[1]), float(point[2]))
    if _MagnumVector3 is not None:
        return _MagnumVector3(*values)
    return list(values)


# --------------------------------------------------------------------------- #
# bake
# --------------------------------------------------------------------------- #
def _navmesh_settings(profile):
    require_habitat()
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    values = profile.navmesh_settings()
    settings.agent_radius = values["agent_radius"]
    settings.agent_height = values["agent_height"]
    settings.agent_max_climb = values["agent_max_climb"]
    settings.agent_max_slope = values["agent_max_slope"]
    settings.cell_size = values["cell_size"]
    settings.cell_height = values["cell_height"]
    settings.include_static_objects = False
    return settings


def _cache_dir(cache_root: str, entry: assets.SceneEntry, fingerprint: str,
               profile) -> str:
    return os.path.join(cache_root, entry.scene_id, fingerprint,
                        profile.navmesh_fingerprint())


def _load_metadata(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _metadata_matches(meta: Optional[dict], floor_id: str, fingerprint: str,
                      profile, source_path: str, source_hash: str) -> bool:
    if not meta:
        return False
    return (meta.get("nav_cache_version") == NAV_CACHE_VERSION
            and meta.get("floor_id") == floor_id
            and meta.get("scene_fingerprint") == fingerprint
            and meta.get("profile_fingerprint") == profile.navmesh_fingerprint()
            and meta.get("source") == source_path
            and meta.get("source_sha256") == source_hash)


def build_floor_navmesh(entry: assets.SceneEntry,
                        spec,
                        floor,
                        profile,
                        cache_root: str,
                        mesh: Optional[assets.Mesh] = None,
                        source_path: Optional[str] = None,
                        source_hash: Optional[str] = None,
                        fingerprint: Optional[str] = None,
                        rebuild: bool = False) -> FloorNavMesh:
    """Bake (or load) one storey's NavMesh for one profile."""
    require_habitat()
    source_path = source_path or assets.resolve_mesh(entry)
    source_hash = source_hash or assets.file_sha256(source_path)
    fingerprint = fingerprint or assets.scene_fingerprint(entry, spec, source_path)
    directory = _cache_dir(cache_root, entry, fingerprint, profile)
    os.makedirs(directory, exist_ok=True)
    stem = str(floor.id or "F0")
    obj_path = os.path.join(directory, f"{stem}.obj")
    navmesh_path = os.path.join(directory, f"{stem}.navmesh")
    meta_path = os.path.join(directory, f"{stem}.json")
    meta = _load_metadata(meta_path)
    start = time.time()

    if not rebuild and os.path.isfile(obj_path) and os.path.isfile(navmesh_path) \
            and _metadata_matches(meta, floor.id, fingerprint, profile,
                                  source_path, source_hash):
        pathfinder = HabitatFloorPathfinder(navmesh_path)
        meta = dict(meta or {})
        meta["cache"] = "hit"
        meta["profile"] = profile.name
        meta["profile_motion_fingerprint"] = profile.fingerprint()
        return FloorNavMesh(floor_id=floor.id, elevation=floor.elevation,
                            height=floor.height, navmesh_path=navmesh_path,
                            collision_obj=obj_path, pathfinder=pathfinder,
                            metadata=meta)

    if mesh is None:
        mesh = assets.read_mesh(source_path)
    floor_mesh = assets.extract_floor_mesh(mesh, spec, floor)
    obj_stats = assets.write_floor_obj(floor_mesh, obj_path)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = os.path.abspath(obj_path)
    sim_cfg.create_renderer = False
    sim_cfg.enable_physics = False
    sim_cfg.load_semantic_mesh = False
    sim_cfg.requires_textures = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        settings = _navmesh_settings(profile)
        ok = simulator.recompute_navmesh(simulator.pathfinder, settings)
        if not ok or not simulator.pathfinder.is_loaded:
            raise RuntimeError(
                f"{entry.scene_id}/{floor.id}: Recast could not build a NavMesh from "
                f"{obj_path} ({obj_stats['triangles']} triangles, profile {profile.name})")
        simulator.pathfinder.save_nav_mesh(navmesh_path)
        num_islands = int(simulator.pathfinder.num_islands)
        area = float(simulator.pathfinder.navigable_area)
    finally:
        simulator.close()

    pathfinder = HabitatFloorPathfinder(navmesh_path)
    meta = {
        "nav_cache_version": NAV_CACHE_VERSION,
        "scene_id": entry.scene_id,
        "scene_fingerprint": fingerprint,
        "profile": profile.name,
        "profile_fingerprint": profile.navmesh_fingerprint(),
        "profile_motion_fingerprint": profile.fingerprint(),
        "navmesh_settings": profile.navmesh_settings(),
        "floor_id": floor.id,
        "elevation": floor.elevation,
        "height": floor.height,
        "source": source_path,
        "source_sha256": source_hash,
        "collision_obj": os.path.basename(obj_path),
        "collision_mesh": obj_stats,
        "num_islands": num_islands,
        "navigable_area": round(area, 4),
        "bake_seconds": round(time.time() - start, 3),
        "cache": "rebuilt",
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return FloorNavMesh(floor_id=floor.id, elevation=floor.elevation,
                        height=floor.height, navmesh_path=navmesh_path,
                        collision_obj=obj_path, pathfinder=pathfinder,
                        metadata=meta)


# --------------------------------------------------------------------------- #
# links from the spec walk tracks
# --------------------------------------------------------------------------- #
def _surface_query(point: Sequence[float], profile) -> Tuple[float, float, float]:
    """Ask for a floor surface at the Recast-quantised height (+ cell_height)."""
    return (float(point[0]), float(point[1]) + profile.cell_height, float(point[2]))


def build_links(spec, profile, floor_navs: Dict[str, FloorNavMesh]) -> List[NavLink]:
    """Turn every spec stair / ramp walk track into a snapped, profile-checked link."""
    links: List[NavLink] = []
    for index, (connector, nodes) in enumerate(spec.walk_tracks()):
        floor_id = assets.floor_id_of_room(spec, connector.from_room)
        to_floor_id = assets.floor_id_of_room(spec, connector.to_room)
        raw_track = [(float(node.point[0]), float(node.z), -float(node.point[1]))
                     for node in nodes]
        link_id = (f"{connector.kind}_{index}_{connector.from_room}_"
                   f"{connector.to_room}")
        provisional = NavLink(
            link_id=link_id, kind=connector.kind,
            from_room=connector.from_room, to_room=connector.to_room,
            from_floor=floor_id, to_floor=to_floor_id,
            from_endpoint=LinkEndpoint(floor_id, raw_track[0], "base"),
            to_endpoint=LinkEndpoint(to_floor_id, raw_track[-1], "head"),
            track=raw_track, connector_index=index,
        )
        problem = link_profile_problem(provisional, profile)
        if problem:
            provisional.valid = False
            provisional.reason = problem
            links.append(provisional)
            continue
        if floor_id not in floor_navs or to_floor_id not in floor_navs:
            provisional.valid = False
            provisional.reason = "floor_navmesh_missing"
            links.append(provisional)
            continue
        base_nav = floor_navs[floor_id]
        head_nav = floor_navs[to_floor_id]
        base_snap = base_nav.snap_point(_surface_query(raw_track[0], profile))
        head_snap = head_nav.snap_point(_surface_query(raw_track[-1], profile))
        tolerance = max(0.75, profile.radius + 0.4)
        if base_snap is None or point_distance(base_snap, raw_track[0]) > tolerance:
            provisional.valid = False
            provisional.reason = "base_gate_off_navmesh"
            links.append(provisional)
            continue
        if head_snap is None or point_distance(head_snap, raw_track[-1]) > tolerance:
            provisional.valid = False
            provisional.reason = "head_gate_off_navmesh"
            links.append(provisional)
            continue
        track = [tuple(base_snap)] + list(raw_track[1:-1]) + [tuple(head_snap)]
        provisional.from_endpoint = LinkEndpoint(floor_id, tuple(base_snap), "base")
        provisional.to_endpoint = LinkEndpoint(to_floor_id, tuple(head_snap), "head")
        provisional.track = track
        links.append(provisional)
    return links


# --------------------------------------------------------------------------- #
# composite
# --------------------------------------------------------------------------- #
class CompositeNavMesh:
    """All storey NavMeshes of one scene plus the stair / ramp links."""

    def __init__(self, entry: assets.SceneEntry, spec, profile,
                 floors: Dict[str, FloorNavMesh], links: List[NavLink],
                 cache_dir: str, fingerprint: str, source_path: str):
        self.entry = entry
        self.spec = spec
        self.profile = profile
        self.floors = floors
        self.links = links
        self.cache_dir = cache_dir
        self.fingerprint = fingerprint
        self.source_path = source_path

    # -- topology ----------------------------------------------------------- #
    def floor(self, floor_id: str) -> FloorNavMesh:
        return self.floors[floor_id]

    @property
    def valid_links(self) -> List[NavLink]:
        return [link for link in self.links if link.valid]

    def find_path(self, start_floor: str, start_point: Sequence[float],
                  goal_floor: str, goal_point: Sequence[float]) -> Optional[CompositePath]:
        return plan_composite_path(self.floors, self.links, start_floor, start_point,
                                   goal_floor, goal_point)

    def is_connected(self, start_floor: str, start_point: Sequence[float],
                     goal_floor: str, goal_point: Sequence[float]) -> bool:
        return self.find_path(start_floor, start_point, goal_floor, goal_point) is not None

    def validate_path(self, path: CompositePath) -> List[PathViolation]:
        return validate_composite_path(self.spec, path, self.profile)

    # -- sampling ----------------------------------------------------------- #
    def snap(self, floor_id: str, point: Sequence[float],
             tolerance: float = 0.75) -> Optional[Tuple[float, float, float]]:
        snapped = self.floors[floor_id].snap_point(point)
        if snapped is None or point_distance(snapped, point) > tolerance:
            return None
        return snapped

    def sample_room_point(self, room_id: str, rng, margin: Optional[float] = None,
                          max_tries: int = 300):
        """A deterministic navigable point inside a room, on its own floor slab.

        The point must clear the walls by ``radius + 0.15``, snap back to within
        0.35 m, stay in the same room polygon and sit at the room's walking
        surface (not on a table or bed, whose NavMesh top is higher).
        """
        from roomgen_nav.pathfinding import sample_point_in_polygon

        room = self.spec.room(room_id)
        floor_id = assets.floor_id_of_room(self.spec, room_id)
        floor_nav = self.floors[floor_id]
        elevation = self.spec.room_elevation(room_id)
        clearance = self.profile.radius + 0.15 if margin is None else margin
        for _ in range(max_tries):
            xy = sample_point_in_polygon(rng, room.polygon, clearance)
            if xy is None:
                continue
            query = (xy[0], elevation + self.profile.cell_height, -xy[1])
            snapped = floor_nav.snap_point(query)
            if snapped is None:
                continue
            if point_distance(snapped, query) > 0.35:
                continue
            if not self.spec_mod_point_in_room((snapped[0], -snapped[2]), room.polygon):
                continue
            if abs(snapped[1] - (elevation + self.profile.cell_height)) > 0.18:
                continue
            return snapped
        return None

    @staticmethod
    def spec_mod_point_in_room(xy, polygon) -> bool:
        from blender_agent import spec as spec_mod
        return bool(spec_mod.point_in_polygon(xy, polygon))

    def sample_start_goal(self, start_room: str, goal_room: str, rng,
                          min_distance: float = 1.5):
        start_floor = assets.floor_id_of_room(self.spec, start_room)
        goal_floor = assets.floor_id_of_room(self.spec, goal_room)
        for _ in range(200):
            start = self.sample_room_point(start_room, rng)
            goal = self.sample_room_point(goal_room, rng)
            if start is None or goal is None:
                continue
            if point_distance(start, goal) < min_distance:
                continue
            path = self.find_path(start_floor, start, goal_floor, goal)
            if path is None:
                continue
            if not path.crosses_floors and start_floor != goal_floor:
                continue
            violations = self.validate_path(path)
            if violations:
                continue
            return start_floor, start, goal_floor, goal, path
        return None

    # -- islands / connected regions --------------------------------------- #
    def components(self) -> Dict[Tuple[str, int], int]:
        """Union-find over (floor, island) nodes; links join their end islands."""
        parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

        def add(node):
            parent.setdefault(node, node)

        def find(node):
            root = node
            while parent[root] != root:
                root = parent[root]
            while parent[node] != root:
                parent[node], node = root, parent[node]
            return root

        def union(a, b):
            add(a)
            add(b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for floor_id, floor_nav in self.floors.items():
            for island in range(floor_nav.num_islands):
                add((floor_id, island))
        for link in self.valid_links:
            island_a = self.floors[link.from_floor].pathfinder.get_island(
                link.from_endpoint.point)
            island_b = self.floors[link.to_floor].pathfinder.get_island(
                link.to_endpoint.point)
            if island_a < 0 or island_b < 0:
                continue
            union((link.from_floor, island_a), (link.to_floor, island_b))
        # Use a compact label derived from the union root so it is stable.
        roots = sorted({find(node) for node in parent}, key=lambda item: (item[0], item[1]))
        labels = {root: index for index, root in enumerate(roots)}
        return {node: labels[find(node)] for node in parent}

    def component_of(self, floor_id: str, point: Sequence[float]) -> Optional[int]:
        island = self.floors[floor_id].pathfinder.get_island(point)
        if island < 0:
            return None
        return self.components().get((floor_id, island))

    def component_summary(self) -> List[dict]:
        mapping = self.components()
        groups: Dict[int, List[dict]] = {}
        for (floor_id, island), component in mapping.items():
            groups.setdefault(component, []).append({
                "floor_id": floor_id,
                "island": island,
                "area": round(self.floors[floor_id].pathfinder.island_area(island), 4),
                "radius": round(self.floors[floor_id].pathfinder.island_radius(island), 4),
            })
        return [{"component": component, "islands": members}
                for component, members in sorted(groups.items())]

    # -- report ------------------------------------------------------------- #
    def summary(self) -> dict:
        return {
            "scene_id": self.entry.scene_id,
            "profile": self.profile.to_dict(),
            "source_mesh": self.source_path,
            "scene_fingerprint": self.fingerprint,
            "cache_dir": self.cache_dir,
            "floors": {floor_id: floor_nav.to_dict()
                       for floor_id, floor_nav in sorted(self.floors.items())},
            "links": [link.to_dict() for link in self.links],
            "valid_links": [link.link_id for link in self.valid_links],
            "components": self.component_summary(),
        }

    def stats(self) -> dict:
        return {
            "floors": len(self.floors),
            "links": len(self.links),
            "valid_links": len(self.valid_links),
            "cross_floor_links": sum(1 for link in self.valid_links if link.is_cross_floor),
            "navigable_area": round(sum(f.navigable_area for f in self.floors.values()), 4),
            "islands": {fid: f.num_islands for fid, f in self.floors.items()},
        }


def build_composite_navmesh(entry: assets.SceneEntry,
                            profile,
                            cache_root: str,
                            rebuild: bool = False,
                            spec=None,
                            mesh: Optional[assets.Mesh] = None) -> CompositeNavMesh:
    """Build / load every storey NavMesh and the stair / ramp links."""
    require_habitat()
    spec = spec or assets.load_scene(entry)
    source_path = assets.resolve_mesh(entry)
    source_hash = assets.file_sha256(source_path)
    fingerprint = assets.scene_fingerprint(entry, spec, source_path)
    if mesh is None:
        mesh = assets.read_mesh(source_path)
    floors: Dict[str, FloorNavMesh] = {}
    for floor in assets.floors_of(spec):
        floors[floor.id] = build_floor_navmesh(
            entry, spec, floor, profile, cache_root,
            mesh=mesh, source_path=source_path, source_hash=source_hash,
            fingerprint=fingerprint, rebuild=rebuild)
    links = build_links(spec, profile, floors)
    return CompositeNavMesh(entry, spec, profile, floors, links,
                            os.path.join(cache_root, entry.scene_id),
                            fingerprint, source_path)
