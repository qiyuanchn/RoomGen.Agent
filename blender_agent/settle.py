"""Physics settling ("drop to rest") for the furniture of a built scene.

What it is for
--------------
``run_build.py`` places furniture analytically: the bottom of an item sits at the
authored ``z`` (0 = "on the floor") and the placement is only cross-checked
against the room outline.  Nothing guarantees that a piece really stands on the
floor slab, that two pieces do not interpenetrate, or that a stack is
consistent.  :func:`ground_furniture` adds the missing physical stage between
:func:`run_build.place_furniture` and the render: every furniture placement gets
a rigid body, the shell (floor / walls / ceiling) becomes a passive collider,
the bodies are dropped a couple of centimetres and simulated until they rest,
and the resulting rigid transform is *baked* into the object transform.

The recipe mirrors BlenderProc's ``simulate_physics_and_fix_final_poses``
("drop to rest"): rigid bodies, ``bpy.ops.ptcache`` frame stepping, then baking
the final pose into the objects -- so nothing of the simulation leaks into the
render (the rigid body world, the proxies and their point cache are removed
again before the cameras are set up).

How it works on this scene
--------------------------
* Furniture models are *collection instances* (an empty that instantiates a
  shared library collection) and Blender's rigid-body operator only accepts mesh
  objects, so each placement gets a **convex-hull proxy**: the convex hull of its
  world-space vertices, simulated as that placement's body.  After the bake the
  proxy's rigid transform ``P`` is composed onto the object,
  ``object.matrix_world = P @ object.matrix_world``, which leaves the instanced
  meshes (and their shared textures) untouched.
* Floor / walls / ceiling are **passive** bodies with ``collision_shape='MESH'``
  (a convex hull would seal the door openings) and a zero collision margin.
* Active bodies carry the **collision margin** (``margin_m``, default 2 mm):
  Bullet resolves contacts on the margin skin, so the margin *is* the resting
  gap -- the manifest's settled ``bbox_z`` bottom is expected to be millimetres,
  never centimetres (Blender's 40 mm default would visibly float everything).
* ``use_split_impulse`` is on: without it a deeply interpenetrating pair is
  ejected (measured on the probe scene: a 0.15 m overlap launches both bodies
  several metres).
* Placements whose hull already intersects another *resting* placement are
  separated **geometrically** first (minimum translation over the hull face
  normals) because Bullet's penetration recovery overshoots by roughly the
  overlap.  In the shipped P1 spec that is the sofa authored 4.6 cm inside the
  bookshelf's hull, and the shell overlaps of the bed, the basket and the
  bedroom plant (6.3 cm into the east wall): they are nudged out, not launched.
* Thin slabs (``builtin:rug``, at most ``flat_height_m`` tall) get their own
  collision group: they rest on the floor like everything else but do not lift
  the furniture standing on them (a 12 mm rug is floor finish, not furniture).
* The hull's **base is flattened** (``base_band_m``, default 1.5 cm) and the
  body's **centre of mass -- the object origin, which is what Bullet takes it
  for -- is placed over that base**, at the hull's volume-centroid height.  A
  pot's base is not planar in the mesh, so its hull would otherwise touch the
  floor in a single vertex, and a piece whose mass hangs outside its footprint
  rolls over by itself (the potted plants' canopy centre is 10 cm off the pot:
  they fell over within a second before both fixes).

What is *not* settled (held placements)
---------------------------------------
A placement is **held** -- its authored transform is kept exactly -- whenever
gravity is not its support:

* ``mount: "ceiling"`` (pendant lamps) and ``mount: "wall"``;
* an authored ``z`` above ``lift_tolerance_m`` (default 2 cm): pictures hung on a
  wall, the book stacks on the shelf, the bedside lamp on the nightstand -- the
  mounted / stacked items the acceptance check exempts.

Held placements still get their active (``kinematic``) rigid body, so *every*
furniture instance takes part in the simulation, but they sit in a private
collision group: they neither fall nor push the resting furniture they overlap
(the books sit *inside* the bookshelf's convex hull).

Configuration
-------------
``PANOWORLD_SETTLE=1|0|auto`` forces settling on or off; ``PANOWORLD_SETTLE_*``
overrides single parameters, and a top-level ``"settle": {...}`` block in the
scene spec does the same per spec (``spec.py`` ignores unknown top-level keys, so
this needs no schema change).  The default is ``auto``: on when the spec places
furniture and is not a ``front3d`` replay -- a P3 replay replays dataset poses,
which must stay exact.

Host side / Blender side
------------------------
Only the pure helpers (configuration, the rest/held classification, the
acceptance maths, the separation maths) live outside Blender; ``bpy`` is imported
lazily inside the functions that need it, so ``tests/test_settle.py`` and the
host tools can import this module.

Self-check (headless)::

    blender --background --factory-startup --python blender_agent/settle.py -- --selfcheck

builds a tiny scene of placed boxes (floating, sunk, overlapping, stacked, on a
rug, in a wall, instanced) and verifies that everything ends up standing on the
floor without penetrating it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- defaults (all overridable, see resolve_config) ------------------------- #
DEFAULT_MARGIN_M = 0.002        # Bullet skin == the resting gap of a settled body
DEFAULT_DROP_M = 0.02           # lift before the drop: proves floor contact, no zero-gap start
DEFAULT_FRAMES = 60             # simulation budget (24 fps)
DEFAULT_CHECK_INTERVAL = 6      # frames between two "is it still moving?" samples
DEFAULT_REST_TOLERANCE_M = 0.0015
DEFAULT_SUBSTEPS = 20
DEFAULT_SOLVER_ITERATIONS = 25
DEFAULT_LIFT_TOLERANCE_M = 0.02  # authored z above this = mounted / stacked -> held
DEFAULT_GROUND_TOLERANCE_M = 0.02  # acceptance: settled bottom within 2 cm of the floor
DEFAULT_PENETRATION_TOLERANCE_M = 0.001
DEFAULT_FLAT_HEIGHT_M = 0.03    # thin slabs are floor finish, not furniture
DEFAULT_BASE_BAND_M = 0.015     # geometry this close to the lowest point makes up the base
DEFAULT_MASS_DENSITY = 120.0    # kg/m^3 of hull volume (furniture is mostly air)
DEFAULT_LINEAR_DAMPING = 0.12   # [m/s per frame-per-second] damped landing, no creep
DEFAULT_ANGULAR_DAMPING = 0.30  # keeps a dropped piece from rocking itself off its base
MIN_MASS_KG = 0.5
MAX_MASS_KG = 400.0
FLOOR_EPS_M = 0.001             # a resting body is pre-lifted to this gap before the drop
SEPARATION_EPS_M = 0.002        # clearance left by the geometric pre-separation
GRAVITY_MS2 = -9.81

#: collision groups (``RigidBodyObject.collision_collections`` bits).  Two bodies
#: interact when their masks share a bit, so the shell needs two bits: one it
#: shares with the furniture and one with the floor coverings, which in turn must
#: not share a bit with each other.
COLLISION_GROUP_SHELL_FURNITURE = 0   # shell <-> resting furniture
COLLISION_GROUP_SHELL_FLAT = 1        # shell <-> floor coverings
COLLISION_GROUP_FURNITURE = 2         # furniture <-> furniture
COLLISION_GROUP_FLAT = 3              # floor covering <-> floor covering
COLLISION_GROUP_HELD = 4              # mounted / stacked: isolated, never interacts
MASK_SHELL = (COLLISION_GROUP_SHELL_FURNITURE, COLLISION_GROUP_SHELL_FLAT)
MASK_FURNITURE = (COLLISION_GROUP_SHELL_FURNITURE, COLLISION_GROUP_FURNITURE)
MASK_FLAT = (COLLISION_GROUP_SHELL_FLAT, COLLISION_GROUP_FLAT)
MASK_HELD = (COLLISION_GROUP_HELD,)

#: collection names (plain strings: importing ``shell``/``front3d`` here would
#: pull in ``bpy`` and break the host-side import of this module).
SHELL_COLLECTION = "shell"
FRONT3D_SHELL_COLLECTION = "front3d_shell"
FURNITURE_COLLECTION = "furniture"
PROXY_COLLECTION = "settle_proxies"

#: spec key this module reads (``spec.py`` ignores unknown top-level keys)
SPEC_BLOCK = "settle"

_ENV_PREFIX = "PANOWORLD_SETTLE"
_PARAMETER_KEYS = (
    "margin_m", "drop_m", "frames", "check_interval", "rest_tolerance_m", "substeps",
    "solver_iterations", "lift_tolerance_m", "ground_tolerance_m",
    "penetration_tolerance_m", "flat_height_m", "base_band_m", "mass_density", "separation",
    "collide_details", "linear_damping", "angular_damping",
)
_ALIASES = {
    "margin": "margin_m", "drop": "drop_m", "rest_tolerance": "rest_tolerance_m",
    "substeps_per_frame": "substeps", "solver_iters": "solver_iterations",
    "lift_tolerance": "lift_tolerance_m", "ground_tolerance": "ground_tolerance_m",
    "flat_height": "flat_height_m", "density": "mass_density", "base_band": "base_band_m",
}
_BOOL_KEYS = ("separation", "collide_details")
_INT_KEYS = ("frames", "check_interval", "substeps", "solver_iterations")


# --------------------------------------------------------------------------- #
# pure: configuration
# --------------------------------------------------------------------------- #
def _flag(value) -> Optional[bool]:
    """Parse an on/off/auto style value (``None`` = auto / not given)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("", "auto", "default", "none"):
        return None
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"cannot read '{value}' as on/off/auto")


def resolve_config(spec=None, environ: Optional[dict] = None, **overrides) -> dict:
    """Effective settling parameters for ``spec`` (spec block < env < overrides).

    Returns the parameter dict plus ``enabled`` (``auto`` is resolved here),
    ``reason``, ``front3d`` and ``notes`` (where every value came from).
    """
    environ = os.environ if environ is None else environ
    block = {}
    if spec is not None:
        raw = getattr(spec, "raw", None) or {}
        candidate = raw.get(SPEC_BLOCK)
        if isinstance(candidate, dict):
            block = dict(candidate)
        elif candidate is not None:
            block = {"enabled": candidate}

    config = {
        "margin_m": DEFAULT_MARGIN_M,
        "drop_m": DEFAULT_DROP_M,
        "frames": DEFAULT_FRAMES,
        "check_interval": DEFAULT_CHECK_INTERVAL,
        "rest_tolerance_m": DEFAULT_REST_TOLERANCE_M,
        "substeps": DEFAULT_SUBSTEPS,
        "solver_iterations": DEFAULT_SOLVER_ITERATIONS,
        "lift_tolerance_m": DEFAULT_LIFT_TOLERANCE_M,
        "ground_tolerance_m": DEFAULT_GROUND_TOLERANCE_M,
        "penetration_tolerance_m": DEFAULT_PENETRATION_TOLERANCE_M,
        "flat_height_m": DEFAULT_FLAT_HEIGHT_M,
        "base_band_m": DEFAULT_BASE_BAND_M,
        "mass_density": DEFAULT_MASS_DENSITY,
        "linear_damping": DEFAULT_LINEAR_DAMPING,
        "angular_damping": DEFAULT_ANGULAR_DAMPING,
        "separation": True,
        "collide_details": False,
    }
    notes: List[str] = []
    enabled: Optional[bool] = None

    def apply(values: dict, source: str) -> None:
        nonlocal enabled
        for key, value in values.items():
            name = _ALIASES.get(key, key)
            if name == "enabled":
                enabled = _flag(value)
                notes.append(f"enabled={enabled} ({source})")
            elif name in config:
                if name in _BOOL_KEYS:
                    config[name] = bool(value)
                elif name in _INT_KEYS:
                    config[name] = int(value)
                else:
                    config[name] = float(value)
                notes.append(f"{name}={config[name]} ({source})")
            else:
                notes.append(f"ignored unknown key '{key}' ({source})")

    apply(block, "spec settle block")
    env_values = {key[len(_ENV_PREFIX) + 1:].lower(): environ[key]
                  for key in environ if key.startswith(_ENV_PREFIX + "_")}
    if environ.get(_ENV_PREFIX) is not None:
        env_values["enabled"] = environ[_ENV_PREFIX]
    apply(env_values, "environment")
    apply(dict(overrides), "caller")

    front3d = bool(getattr(spec, "front3d", None))
    if enabled is None:
        has_furniture = bool(getattr(spec, "furniture", None))
        enabled = has_furniture and not front3d
        if front3d:
            reason = "front3d replay: dataset poses are exact and stay untouched"
        elif not has_furniture:
            reason = "no furniture in the spec"
        else:
            reason = "auto: furniture is placed, nothing marks it as mounted"
    else:
        reason = "forced by the spec settle block / environment / caller"

    for key in ("margin_m", "drop_m", "rest_tolerance_m", "lift_tolerance_m", "ground_tolerance_m",
                "penetration_tolerance_m", "flat_height_m", "base_band_m", "mass_density",
                "linear_damping", "angular_damping"):
        if config[key] < 0.0:
            raise ValueError(f"{key} must be >= 0")
    if config["frames"] < 1:
        raise ValueError("frames must be >= 1")
    if config["substeps"] < 1 or config["solver_iterations"] < 1:
        raise ValueError("substeps / solver_iterations must be >= 1")

    config["enabled"] = bool(enabled)
    config["front3d"] = front3d
    config["reason"] = reason
    config["notes"] = notes
    return config


# --------------------------------------------------------------------------- #
# pure: what rests, what is held
# --------------------------------------------------------------------------- #
def placement_mode(item, placement: dict, config: dict) -> Tuple[str, str]:
    """``("rest" | "held", reason)`` for one placement.

    ``item`` is the spec's :class:`~blender_agent.spec.FurnitureSpec` (``None``
    for a placement the spec does not describe) and ``placement`` is the matching
    entry of ``furniture_info["placements"]``.
    """
    mount = getattr(item, "mount", None) or placement.get("mount") or "floor"
    z = getattr(item, "z", None)
    if z is None:
        z = placement.get("z")
    lift = float(config["lift_tolerance_m"])
    if mount == "ceiling":
        return "held", "mount: ceiling (the lamp hangs from the ceiling)"
    if mount == "wall":
        return "held", "mount: wall (the piece is fixed to a wall)"
    if z is not None and float(z) > lift:
        return "held", f"authored z={float(z):.3f} m (mounted / stacked, exempt from the floor check)"
    return "rest", "rests on the floor"


def plan(spec, placements: Sequence[dict], config: dict) -> List[dict]:
    """Rest/held decision for every placement, in spec order."""
    items = list(getattr(spec, "furniture", None) or [])
    decisions = []
    for index, placement in enumerate(placements):
        item = items[index] if index < len(items) else None
        mode, reason = placement_mode(item, placement, config)
        decisions.append({
            "index": index,
            "name": placement.get("name") or f"item_{index}",
            "asset": placement.get("asset"),
            "mode": mode,
            "reason": reason,
        })
    return decisions


def ground_check(bottom_m: float, config: dict) -> dict:
    """Acceptance maths for one settled bottom height (metres, floor at 0)."""
    ground = float(config["ground_tolerance_m"])
    penetration = float(config["penetration_tolerance_m"])
    return {
        "bottom_m": round(float(bottom_m), 5),
        "gap_m": round(float(bottom_m), 5),
        "grounded": -penetration <= float(bottom_m) <= ground,
        "penetrating": float(bottom_m) < -penetration,
        "floating_m": round(max(0.0, float(bottom_m) - ground), 5),
    }


def bbox_z(points) -> Optional[List[float]]:
    """``[z_min, z_max]`` of an ``(N, 3)`` point array (``None`` when empty)."""
    if points is None or len(points) == 0:
        return None
    return [round(float(points[:, 2].min()), 4), round(float(points[:, 2].max()), 4)]


# --------------------------------------------------------------------------- #
# pure: geometric pre-separation (numpy, driven from the Blender side)
# --------------------------------------------------------------------------- #
def minimum_translation(a_points, b_points, directions, eps: float = SEPARATION_EPS_M):
    """Separate hull ``b`` from hull ``a`` with the least motion.

    ``directions`` are candidate unit vectors (the hull face normals of both
    bodies).  For a convex body the free translation along ``d`` is
    ``max(a . d) - min(b . d)``; the smallest value over the candidates is the
    minimum translation vector of the pair (exact for face-face contacts, a tight
    upper bound otherwise).  Returns ``(direction, distance)`` or ``(None, 0.0)``
    when the candidate set already separates the hulls.
    """
    import numpy as np

    directions = np.asarray(directions, dtype=np.float64)
    projections_a = np.asarray(a_points, dtype=np.float64) @ directions.T
    projections_b = np.asarray(b_points, dtype=np.float64) @ directions.T
    gaps = projections_a.max(axis=0) - projections_b.min(axis=0)
    index = int(np.argmin(gaps))
    if gaps[index] <= 0.0:
        return None, 0.0
    return directions[index], float(gaps[index]) + float(eps)


# --------------------------------------------------------------------------- #
# Blender side
# --------------------------------------------------------------------------- #
def _bpy():
    import bpy
    return bpy


def _matrix_array(matrix):
    import numpy as np
    return np.array([tuple(row) for row in matrix], dtype=np.float64)


def _mesh_points(mesh, matrix):
    import numpy as np
    count = len(mesh.vertices)
    if count == 0:
        return np.zeros((0, 3), dtype=np.float64)
    coords = np.empty(count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(count, 3)
    matrix = _matrix_array(matrix)
    return coords @ matrix[:3, :3].T + matrix[:3, 3]


def collection_meshes(collection, _seen=None) -> List:
    """Every mesh object of a collection tree, de-duplicated."""
    if collection is None:
        return []
    seen = set() if _seen is None else _seen
    out = []
    for obj in collection.objects:
        if obj.name in seen:
            continue
        seen.add(obj.name)
        if obj.type == "MESH":
            out.append(obj)
    for child in collection.children:
        out += collection_meshes(child, seen)
    return out


def _local_matrix(obj):
    """``obj``'s matrix relative to the top of its parent chain (no depsgraph)."""
    matrix = obj.matrix_basis.copy()
    parent = obj.parent
    while parent is not None:
        matrix = parent.matrix_basis @ matrix
        parent = parent.parent
    return matrix


def object_world_points(obj):
    """World-space vertices of ``obj``, with collection instances expanded.

    Library collections are usually *not* linked to the scene, so their children
    never show up in the depsgraph on their own; the instance empty's transform is
    composed with each child's local matrix instead -- the same geometry the PLY
    export walks.
    """
    import numpy as np

    if obj.type == "EMPTY" and obj.instance_type == "COLLECTION" and obj.instance_collection is not None:
        chunks = [
            _mesh_points(child.data, obj.matrix_world @ _local_matrix(child))
            for child in collection_meshes(obj.instance_collection)
        ]
        chunks = [chunk for chunk in chunks if len(chunk)]
        return np.vstack(chunks) if chunks else np.zeros((0, 3), dtype=np.float64)
    if obj.type == "MESH":
        return _mesh_points(obj.data, obj.matrix_world)
    return np.zeros((0, 3), dtype=np.float64)


def _select_only(objects) -> None:
    bpy = _bpy()
    view_layer = bpy.context.view_layer
    for obj in view_layer.objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    view_layer.objects.active = objects[0] if objects else None


def _placement_points(objects, np):
    """World-space vertices of every object of one placement, stacked."""
    chunks = [chunk for chunk in (object_world_points(obj) for obj in objects) if len(chunk)]
    if not chunks:
        return None
    return np.vstack(chunks) if len(chunks) > 1 else chunks[0]


def _ensure_collection(name: str):
    bpy = _bpy()
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def proxy_world_points(proxy):
    """World-space hull vertices of a proxy (the mesh sits around its own origin)."""
    return _mesh_points(proxy.data, proxy.matrix_world)


def flatten_base(points, band_m: float):
    """Give the hull a flat *base*: the floor contact patch of the geometry.

    A potted plant's pot base is not planar in the mesh (polygonal approximation),
    so its convex hull touches the floor in a single vertex and the piece tips
    over as soon as it is dropped.  The geometry within ``band_m`` of the lowest
    point is the piece's footprint; projecting it onto the lowest plane makes the
    hull stand on that patch instead of on a point -- the height does not change,
    only the support polygon.
    """
    import numpy as np

    if band_m <= 0.0 or len(points) == 0:
        return points
    lowest = float(points[:, 2].min())
    near = points[points[:, 2] <= lowest + band_m]
    if len(near) < 3:
        return points
    footprint = np.array(near, dtype=np.float64)
    footprint[:, 2] = lowest
    return np.vstack([points, footprint])


def mesh_volume_centroid(bm):
    """Volume centroid of a closed mesh, by summing tetrahedra from the origin."""
    import numpy as np

    volume = 0.0
    moment = np.zeros(3)
    for face in bm.faces:
        verts = [np.array(vert.co) for vert in face.verts]
        for index in range(1, len(verts) - 1):
            a, b, c = verts[0], verts[index], verts[index + 1]
            tetra = float(np.dot(a, np.cross(b, c))) / 6.0
            volume += tetra
            moment += tetra * (a + b + c) / 4.0
    if abs(volume) < 1e-12:
        return np.zeros(3)
    return moment / volume


def _build_hull(name: str, points, collection, bpy, bmesh):
    """Convex-hull proxy mesh of a world-space point cloud.

    The mesh data is centred on the object's origin and the origin carries the
    body's *centre of mass*, because Bullet takes the object origin for it:

    * the origin has to be **inside the shape** -- a body whose geometry is far
      from its origin tumbles and drops through a passive floor mesh (measured on
      the probe scene);
    * it has to be **over the support patch**.  A piece whose mass hangs outside
      its footprint tips over as soon as it is dropped: the potted plants' canopy
      centre is ~10 cm off the pot, and they rolled over within a second.  The
      origin is therefore placed at the xy centre of the base patch, at the height
      of the hull's volume centroid -- the mass stays where it belongs
      vertically, but it no longer hangs beyond the base.

    The world pose at creation is therefore ``T(origin)``, and the net rigid
    motion of the body is ``final @ start^-1``.  Returns ``(object, volume_m3)``;
    ``object`` is ``None`` when the point cloud is degenerate.
    """
    import numpy as np
    from mathutils import Vector

    bm = bmesh.new()
    for point in points:
        bm.verts.new((float(point[0]), float(point[1]), float(point[2])))
    mesh = None
    try:
        bmesh.ops.convex_hull(bm, input=bm.verts, use_existing_faces=False)
        loose = [vert for vert in bm.verts if not vert.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        volume = abs(float(bm.calc_volume(signed=True))) if bm.faces else 0.0
        centroid = mesh_volume_centroid(bm) if bm.faces else np.zeros(3)
        lowest = min((vert.co.z for vert in bm.verts), default=0.0)
        base = np.array([tuple(vert.co) for vert in bm.verts if vert.co.z <= lowest + 1e-6])
        base_center = base.mean(axis=0) if len(base) else centroid
        origin = np.array([base_center[0], base_center[1], centroid[2]])
        for vert in bm.verts:
            vert.co = vert.co - Vector(tuple(float(v) for v in origin))
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    if mesh is None or not mesh.polygons:
        if mesh is not None:
            bpy.data.meshes.remove(mesh)
        return None, 0.0
    obj = bpy.data.objects.new(name, mesh)
    obj.location = (float(origin[0]), float(origin[1]), float(origin[2]))
    collection.objects.link(obj)
    return obj, volume


def _mesh_normals(mesh):
    import numpy as np
    normals = np.empty(len(mesh.polygons) * 3, dtype=np.float64)
    mesh.polygons.foreach_get("normal", normals)
    return normals.reshape(-1, 3)


def _mask_bits(groups) -> List[bool]:
    bits = [False] * 20
    for group in groups:
        bits[group] = True
    return bits


def _add_body(obj, *, active: bool, margin: float, shape: str, groups, mass: Optional[float] = None,
              kinematic: bool = False, linear_damping: float = None,
              angular_damping: float = None) -> None:
    """Give ``obj`` a rigid body (the operator only accepts mesh objects)."""
    bpy = _bpy()
    _select_only([obj])
    bpy.ops.rigidbody.object_add()
    body = obj.rigid_body
    body.type = "ACTIVE" if active else "PASSIVE"
    body.collision_shape = shape
    body.collision_margin = float(margin)
    body.collision_collections = _mask_bits(groups)
    body.friction = 0.6
    body.use_deactivation = True
    if kinematic:
        body.kinematic = True
    if mass is not None:
        body.mass = float(mass)
    if linear_damping is not None:
        body.linear_damping = float(linear_damping)
    if angular_damping is not None:
        body.angular_damping = float(angular_damping)


def _hull_tree(hull: dict):
    """BVH tree of a hull proxy, in world space (the proxy's mesh is local)."""
    from mathutils.bvhtree import BVHTree
    points = hull["points"]
    vertices = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    if len(vertices) != len(hull["proxy"].data.vertices):
        points = proxy_world_points(hull["proxy"])
        vertices = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    polygons = [list(polygon.vertices) for polygon in hull["proxy"].data.polygons]
    return BVHTree.FromPolygons(vertices, polygons, all_triangles=False)


def _hulls_overlap(hull_a: dict, hull_b: dict, tree_a, tree_b) -> bool:
    """True when two hull proxies actually intersect (not only their AABBs)."""
    for axis in range(3):
        if hull_a["points"][:, axis].min() > hull_b["points"][:, axis].max():
            return False
        if hull_b["points"][:, axis].min() > hull_a["points"][:, axis].max():
            return False
    return bool(tree_a.overlap(tree_b))


def _object_tree(obj):
    """BVH tree of a real object's mesh in world space."""
    from mathutils.bvhtree import BVHTree
    points = _mesh_points(obj.data, obj.matrix_world)
    vertices = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    polygons = [list(polygon.vertices) for polygon in obj.data.polygons]
    return BVHTree.FromPolygons(vertices, polygons, all_triangles=False)


def _separate_static(hulls: List[dict], passive: List, config: dict, notes: List[str]) -> List[dict]:
    """Push resting hulls out of the shell they were authored into.

    A piece modelled a few centimetres into a wall would otherwise be *launched*
    by Bullet's penetration recovery (measured: a 6 cm overlap threw the body
    1.3 m across the room).  Detection is exact (hull BVH against the shell mesh
    BVH), the correction is the minimum translation over the convex hulls.
    """
    import numpy as np
    from mathutils import Vector

    separations: List[dict] = []
    if not passive:
        return separations
    trees = {obj.name: _object_tree(obj) for obj in passive}
    for hull in hulls:
        for obj in passive:
            shell_points = _mesh_points(obj.data, obj.matrix_world)
            if len(shell_points) == 0 or not _hulls_overlap(
                    {"points": hull["points"]}, {"points": shell_points},
                    _hull_tree(hull), trees[obj.name]):
                continue
            directions = np.vstack([
                _mesh_normals(hull["proxy"].data), _mesh_normals(obj.data),
                (hull["center"] - shell_points.mean(axis=0))[None, :],
            ])
            directions = np.vstack([directions, -directions])
            lengths = np.linalg.norm(directions, axis=1)
            keep = lengths > 1e-9
            directions = directions[keep] / lengths[keep][:, None]
            direction, distance = minimum_translation(
                shell_points, hull["points"], directions, eps=float(SEPARATION_EPS_M)
            )
            if direction is None:
                continue
            offset = np.asarray(direction, dtype=np.float64) * distance
            matrix = hull["proxy"].matrix_world.copy()
            matrix.translation = matrix.translation + Vector(tuple(float(v) for v in offset))
            hull["proxy"].matrix_world = matrix
            _bpy().context.view_layer.update()
            hull["points"] = proxy_world_points(hull["proxy"])
            hull["center"] = hull["points"].mean(axis=0)
            hull["separation_m"] += float(distance)
            separations.append({
                "a": obj.name, "b": hull["name"], "moved": hull["name"], "static": True,
                "distance_m": round(float(distance), 4),
                "direction": [round(float(v), 4) for v in direction],
            })
    notes.extend(
        f"pushed {item['b']} out of {item['a']} by {item['distance_m'] * 1000:.1f} mm"
        for item in separations
    )
    return separations


def _masks_interact(a, b) -> bool:
    """True when two bodies' collision masks share a bit (i.e. they collide)."""
    return any(bit in b for bit in a)


def _separate(hulls: List[dict], config: dict, notes: List[str]) -> List[dict]:
    """Geometric pre-separation of interpenetrating *resting* hulls.

    Returns ``[{"a", "b", "moved", "distance_m", "direction"}]`` and leaves the
    moved bodies' proxies translated (the bake picks the offset up like any other
    rigid transform).
    """
    import numpy as np
    from mathutils import Vector

    separations: List[dict] = []
    if not config["separation"] or len(hulls) < 2:
        return separations
    for hull in hulls:
        hull["points"] = proxy_world_points(hull["proxy"])
        hull["center"] = hull["points"].mean(axis=0)
    trees = {hull["name"]: _hull_tree(hull) for hull in hulls}
    for _pass in range(3):
        moved_any = False
        for index, hull_a in enumerate(hulls):
            for hull_b in hulls[index + 1:]:
                if not _masks_interact(hull_a["groups"], hull_b["groups"]):
                    continue  # e.g. a floor covering does not collide with furniture
                if not _hulls_overlap(hull_a, hull_b, trees[hull_a["name"]], trees[hull_b["name"]]):
                    continue
                # the lighter body moves, the heavier one is the anchor
                anchor, mobile = (hull_a, hull_b) if hull_a["mass"] >= hull_b["mass"] else (hull_b, hull_a)
                directions = np.vstack([
                    _mesh_normals(anchor["proxy"].data),
                    _mesh_normals(mobile["proxy"].data),
                    (mobile["center"] - anchor["center"])[None, :],
                ])
                directions = np.vstack([directions, -directions])
                lengths = np.linalg.norm(directions, axis=1)
                keep = lengths > 1e-9
                directions = directions[keep] / lengths[keep][:, None]
                direction, distance = minimum_translation(
                    anchor["points"], mobile["points"], directions, eps=float(SEPARATION_EPS_M)
                )
                if direction is None:
                    notes.append(f"{anchor['name']} <-> {mobile['name']}: overlapping, no clean axis found")
                    continue
                offset = np.asarray(direction, dtype=np.float64) * distance
                matrix = mobile["proxy"].matrix_world.copy()
                matrix.translation = matrix.translation + Vector(tuple(float(v) for v in offset))
                mobile["proxy"].matrix_world = matrix
                _bpy().context.view_layer.update()
                mobile["points"] = proxy_world_points(mobile["proxy"])
                mobile["center"] = mobile["points"].mean(axis=0)
                mobile["separation_m"] += float(distance)
                trees[mobile["name"]] = _hull_tree(mobile)
                separations.append({
                    "a": anchor["name"], "b": mobile["name"], "moved": mobile["name"],
                    "distance_m": round(float(distance), 4),
                    "direction": [round(float(v), 4) for v in direction],
                })
                moved_any = True
        if not moved_any:
            break
    notes.extend(
        f"pre-separated {item['a']} <-> {item['b']}: {item['moved']} moved "
        f"{item['distance_m'] * 1000:.1f} mm"
        for item in separations
    )
    return separations


def _simulate(scene, hulls: List[dict], config: dict) -> dict:
    """Step the point cache until every moving body is at rest (or the budget ends)."""
    frames = int(config["frames"])
    interval = max(1, int(config["check_interval"]))
    tolerance = float(config["rest_tolerance_m"])
    tracked = [hull for hull in hulls if not hull["kinematic"]]
    window: Dict[str, Tuple[float, float, float]] = {}
    used = 0
    settled = None
    peak_motion = None
    for frame in range(1, frames + 1):
        scene.frame_set(frame)
        used = frame
        if frame % interval:
            continue
        depsgraph = _bpy().context.evaluated_depsgraph_get()
        positions = {
            hull["name"]: tuple(hull["proxy"].evaluated_get(depsgraph).matrix_world.translation)
            for hull in tracked
        }
        if window:
            motion = max(
                max(abs(positions[name][axis] - window[name][axis]) for axis in range(3))
                for name in positions
            )
            peak_motion = motion if peak_motion is None else max(peak_motion, motion)
            if motion <= tolerance:
                settled = frame
                break
        window = positions
    return {
        "frames_used": used,
        "settled_frame": settled,
        "at_rest": settled is not None,
        "rest_tolerance_m": tolerance,
        "peak_motion_m": None if peak_motion is None else round(float(peak_motion), 5),
        "bodies": len(hulls),
        "simulated_bodies": len(tracked),
    }


def _bake(hulls: List[dict], notes: List[str]) -> None:
    """Write every body's final pose into its object, then drop the bodies."""
    bpy = _bpy()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    finals = {hull["name"]: hull["proxy"].evaluated_get(depsgraph).matrix_world.copy() for hull in hulls}
    # read the final pose first: removing the bodies rewinds the objects to their
    # pre-simulation transforms, and the pose is composed onto those
    _select_only([hull["proxy"] for hull in hulls])
    bpy.ops.rigidbody.objects_remove()
    moved = 0
    for hull in hulls:
        final = finals[hull["name"]]
        # the proxy's frame is offset by its centre, so the net rigid motion of the
        # furniture geometry is ``final @ start^-1``
        rigid = final @ hull["proxy_start_inverse"]
        hull["rotation_deg"] = round(float(rigid.to_quaternion().angle) * 180.0 / math.pi, 3)
        first = hull["object"]
        hull["moved_m"] = round(
            float(((rigid @ hull["matrix_before"][first.name]).translation
                   - hull["matrix_before"][first.name].translation).length), 5
        )
        for obj in hull["object_list"]:
            pose = rigid @ hull["matrix_before"][obj.name]
            before = obj.matrix_world
            if max(abs(pose[row][column] - before[row][column])
                   for row in range(4) for column in range(4)) > 1e-9:
                obj.matrix_world = pose
        moved += 1
    notes.append(f"baked {moved} object transforms")


def _teardown(bpy, proxies: List, collection) -> None:
    """Remove the proxies, their meshes, the proxy collection and the sim world."""
    for proxy in proxies:
        mesh = proxy.data
        bpy.data.objects.remove(proxy, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    if collection is not None and not collection.objects and not collection.children:
        bpy.data.collections.remove(collection)
    try:
        bpy.ops.ptcache.free_bake_all()
    except Exception:  # noqa: BLE001 - nothing cached is not a failure
        pass
    try:
        bpy.ops.rigidbody.world_remove()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# entry point used by run_build.py
# --------------------------------------------------------------------------- #
def resolve_targets(placements: Sequence[dict]) -> List[dict]:
    """Map placements to the scene objects that carry their geometry.

    ``place_furniture`` records the instance empty of a model as ``instance``; the
    generated ``builtin:`` props (rug, table lamp) are plain meshes named
    ``rug_<name>`` / ``lamp_<name>_*`` in the furniture collection.
    """
    bpy = _bpy()
    furniture = bpy.data.collections.get(FURNITURE_COLLECTION)
    names = sorted(obj.name for obj in collection_meshes(furniture))
    targets = []
    for index, placement in enumerate(placements):
        label = placement.get("name") or f"item_{index}"
        explicit = placement.get("objects")
        instance = placement.get("instance")
        objects = []
        if explicit:
            objects = [bpy.data.objects.get(name) for name in explicit]
        elif instance:
            objects = [bpy.data.objects.get(instance)]
        elif placement.get("kind") == "builtin" and str(placement.get("asset", "")).startswith("builtin:"):
            prop = str(placement["asset"]).split(":", 1)[1]
            prefix = "rug_" if prop == "rug" else f"lamp_{label}"
            objects = [bpy.data.objects[name] for name in names if name.startswith(prefix)]
        targets.append({"index": index, "name": label, "objects": [obj for obj in objects if obj is not None]})
    return targets


def ground_furniture(spec, furniture_info: dict, build_dir: Optional[str] = None,
                     shell_info: Optional[dict] = None, **overrides) -> dict:
    """Drop the furniture of a built scene to rest and bake the result.

    ``furniture_info`` is the dict :func:`run_build.place_furniture` returns; the
    placement entries are updated in place (``bbox_z``, ``z`` and a ``settle``
    block) so the manifest describes the settled scene.  Returns the report that
    is also written to ``<build_dir>/settle_report.json``.
    """
    import numpy as np

    started = time.time()
    config = resolve_config(spec, **overrides)
    placements = list(furniture_info.get("placements") or [])
    report: dict = {
        "kind": "blender_agent_settle",
        "enabled": bool(config["enabled"]),
        "reason": config.get("reason"),
        "config": {key: config[key] for key in _PARAMETER_KEYS},
        "notes": list(config["notes"]),
        "objects": [],
        "held": [],
        "separations": [],
        "checks": {},
    }
    if not placements:
        report["notes"].append("no furniture placements: nothing to settle")
        return _finish(report, build_dir, started)
    if not config["enabled"]:
        report["notes"].append(f"settling disabled ({config.get('reason')})")
        print(f"[settle] disabled: {config.get('reason')}")
        return _finish(report, build_dir, started)

    bpy = _bpy()
    import bmesh
    from mathutils import Vector

    scene = bpy.context.scene
    # a just-linked object's ``matrix_world`` is stale until the depsgraph has run:
    # the placement sets ``location`` and the geometry would be read from the old
    # pose (an instance empty that moved would be simulated at its old place)
    bpy.context.view_layer.update()
    decisions = plan(spec, placements, config)
    targets = resolve_targets(placements)
    proxies: List = []
    hulls: List[dict] = []
    collection = _ensure_collection(PROXY_COLLECTION)
    missing: List[str] = []

    for decision, target in zip(decisions, targets):
        if not target["objects"]:
            missing.append(decision["name"])
            continue
        points = None
        for obj in target["objects"]:
            chunk = object_world_points(obj)
            if len(chunk):
                points = chunk if points is None else np.vstack([points, chunk])
        if points is None or len(points) < 4:
            missing.append(decision["name"])
            continue
        proxy, volume = _build_hull(f"settle_{decision['name']}",
                                    flatten_base(points, float(config["base_band_m"])),
                                    collection, bpy, bmesh)
        if proxy is None:
            missing.append(decision["name"])
            continue
        proxies.append(proxy)
        height = float(points[:, 2].max() - points[:, 2].min())
        flat = bool(decision["mode"] == "rest" and height <= float(config["flat_height_m"]))
        hulls.append({
            "name": decision["name"],
            "asset": decision["asset"],
            "mode": decision["mode"],
            "reason": decision["reason"],
            "object": target["objects"][0],
            "object_list": list(target["objects"]),
            "objects": [obj.name for obj in target["objects"]],
            "proxy": proxy,
            "geometry": points,
            "bbox_z_before": bbox_z(points),
            "mass": min(max(volume * float(config["mass_density"]), MIN_MASS_KG), MAX_MASS_KG),
            "kinematic": decision["mode"] == "held",
            "flat": flat,
            "groups": (MASK_HELD if decision["mode"] == "held"
                       else (MASK_FLAT if flat else MASK_FURNITURE)),
            "separation_m": 0.0,
            "lift_m": 0.0,
        })
    if missing:
        report["notes"].append(f"no geometry found for: {', '.join(sorted(missing))}")

    # a freshly linked object's ``matrix_world`` is stale until the view layer has
    # been evaluated -- the proxy frames and the pre-settling poses are read here
    bpy.context.view_layer.update()
    for hull in hulls:
        proxy = hull["proxy"]
        hull["points"] = proxy_world_points(proxy)
        hull["center"] = hull["points"].mean(axis=0)
        hull["proxy_start_inverse"] = proxy.matrix_world.inverted()
        hull["matrix_before"] = {obj.name: obj.matrix_world.copy() for obj in hull["object_list"]}

    if not hulls:
        report["notes"].append("nothing to simulate")
        _teardown(bpy, proxies, collection)
        return _finish(report, build_dir, started)

    passive = []
    if shell_info:
        passive += list(shell_info.get("shell_objects") or [])
        if config["collide_details"]:
            passive += list(shell_info.get("details_objects") or [])
    if not passive:
        for name in (SHELL_COLLECTION, FRONT3D_SHELL_COLLECTION):
            passive += collection_meshes(bpy.data.collections.get(name))
    proxy_names = {hull["proxy"].name for hull in hulls}
    passive = [obj for obj in passive if obj.type == "MESH" and obj.name not in proxy_names]

    # --- pose the bodies *before* they exist ------------------------------ #
    # Bullet snapshots an object's pose when the rigid body is created, so the
    # drop lift and the pre-separation have to happen first (otherwise every body
    # starts inside whatever it was authored into).
    for hull in hulls:
        proxy = hull["proxy"]
        lift = max(0.0, FLOOR_EPS_M - float(hull["points"][:, 2].min()))
        if hull["mode"] == "rest":
            lift += float(config["drop_m"])
        if lift > 0.0:
            matrix = proxy.matrix_world.copy()
            matrix.translation = matrix.translation + Vector((0.0, 0.0, lift))
            proxy.matrix_world = matrix
            hull["lift_m"] = round(float(lift), 5)
    bpy.context.view_layer.update()
    for hull in hulls:
        hull["points"] = proxy_world_points(hull["proxy"])
        hull["center"] = hull["points"].mean(axis=0)

    resting = [hull for hull in hulls if not hull["kinematic"]]
    report["separations"] = _separate(resting, config, report["notes"])
    if config["separation"]:
        report["separations"] += _separate_static(resting, passive, config, report["notes"])

    # --- rigid body world, passive shell, furniture bodies ---------------- #
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    world = scene.rigidbody_world
    world.substeps_per_frame = int(config["substeps"])
    world.solver_iterations = int(config["solver_iterations"])
    world.use_split_impulse = True
    world.point_cache.frame_start = 1
    world.point_cache.frame_end = int(config["frames"]) + 1
    previous_gravity = tuple(scene.gravity)
    scene.gravity = (0.0, 0.0, GRAVITY_MS2)

    for obj in passive:
        _add_body(obj, active=False, margin=0.0, shape="MESH", groups=MASK_SHELL)
    report["shell_bodies"] = len(passive)

    for hull in hulls:
        groups = MASK_HELD if hull["kinematic"] else (MASK_FLAT if hull["flat"] else MASK_FURNITURE)
        _add_body(hull["proxy"], active=True, margin=float(config["margin_m"]), shape="CONVEX_HULL",
                  groups=groups, mass=hull["mass"], kinematic=hull["kinematic"],
                  linear_damping=float(config["linear_damping"]),
                  angular_damping=float(config["angular_damping"]))

    simulation = _simulate(scene, hulls, config)
    report["simulation"] = simulation
    _bake(hulls, report["notes"])
    scene.gravity = previous_gravity

    # --- measurements, report, placement bookkeeping ----------------------- #
    geometry = {hull["name"]: _placement_points(hull["object_list"], np) for hull in hulls}
    contents = []
    for hull in hulls:
        after = bbox_z(geometry[hull["name"]])
        entry = {
            "name": hull["name"],
            "asset": hull["asset"],
            "mode": hull["mode"],
            "reason": hull["reason"],
            "objects": hull["objects"],
            "mass_kg": round(float(hull["mass"]), 2),
            "flat": hull["flat"],
            "lift_m": hull["lift_m"],
            "separation_m": round(float(hull["separation_m"]), 4),
            "bbox_z_before": hull["bbox_z_before"],
            "bbox_z_after": after,
            "moved_m": hull.get("moved_m", 0.0),
            "rotation_deg": hull.get("rotation_deg", 0.0),
        }
        if hull["mode"] == "rest" and after is not None:
            entry.update(ground_check(after[0], config))
            # a placement that came to rest *on top of another one* (a stack) is
            # not supposed to touch the floor -- the same exemption the acceptance
            # check makes for the mounted / shelved items
            entry["support"] = _support_of(hull["name"], geometry, config)
            if entry["support"]:
                entry["grounded"] = True
        contents.append(entry)
    report["objects"] = contents
    report["held"] = [entry for entry in contents if entry["mode"] == "held"]
    report["checks"] = _checks(contents, config)
    _update_placements(placements, contents, config)
    _sync_lamp_lights(spec, placements, furniture_info.get("lamp_lights") or [], hulls)

    _teardown(bpy, proxies, collection)
    report["seconds"] = round(time.time() - started, 2)
    print(
        f"[settle] {report['checks']['resting']} resting / {len(report['held'])} held placements, "
        f"{len(report['separations'])} pre-separated, "
        f"{simulation['frames_used']} frames"
        f"{' (at rest)' if simulation['at_rest'] else ' (STILL MOVING)'}, "
        f"bottom gap {report['checks']['max_bottom_gap_m']} m, "
        f"{'; '.join(report['checks']['violations']) or 'no violations'}"
    )
    return _finish(report, build_dir, started)


def _support_of(name: str, geometry: dict, config: dict) -> Optional[str]:
    """Name of the placement ``name`` rests on, if any (a stack, not the floor)."""
    import numpy as np

    points = geometry.get(name)
    if points is None or len(points) == 0:
        return None
    tolerance = float(config["ground_tolerance_m"])
    penetration = float(config["penetration_tolerance_m"])
    low = points.min(axis=0)
    for other, other_points in geometry.items():
        if other == name or other_points is None or len(other_points) == 0:
            continue
        other_high = other_points.max(axis=0)
        # rests *on top of* the other placement: within the tolerance above its
        # top surface (not inside it, and not merely nearby)
        if other_high[2] - penetration > low[2] or other_high[2] < low[2] - tolerance:
            continue
        overlap_x = min(points[:, 0].max(), other_points[:, 0].max()) - max(points[:, 0].min(),
                                                                          other_points[:, 0].min())
        overlap_y = min(points[:, 1].max(), other_points[:, 1].max()) - max(points[:, 1].min(),
                                                                          other_points[:, 1].min())
        if overlap_x > 0.02 and overlap_y > 0.02:
            return other
    return None


def _checks(contents: List[dict], config: dict) -> dict:
    """Acceptance summary: every resting placement hugs the floor, none sinks.

    Placements that end up *stacked* on another placement (``support``) are
    exempt from the floor check, exactly like the mounted / shelved items the
    acceptance exempts -- and they are listed separately.
    """
    resting = [entry for entry in contents if entry["mode"] == "rest"]
    held = [entry for entry in contents if entry["mode"] == "held"]
    stacked = [entry for entry in resting if entry.get("support")]
    violations = []
    for entry in resting:
        if entry["bbox_z_after"] is None:
            violations.append(f"{entry['name']}: no geometry")
        elif entry in stacked:
            continue
        elif entry["penetrating"]:
            violations.append(f"{entry['name']}: sunk {abs(entry['bottom_m']) * 1000:.1f} mm into the floor")
        elif not entry["grounded"]:
            violations.append(f"{entry['name']}: rests {entry['bottom_m'] * 1000:.1f} mm above the floor")
    bottoms = [entry["bottom_m"] for entry in resting if "bottom_m" in entry and entry not in stacked]
    return {
        "resting": len(resting),
        "on_floor": len(resting) - len(stacked),
        "stacked": [{"name": entry["name"], "support": entry["support"],
                     "bottom_m": entry["bottom_m"]} for entry in stacked],
        "held": len(held),
        "max_bottom_gap_m": round(max(bottoms), 5) if bottoms else None,
        "min_bottom_m": round(min(bottoms), 5) if bottoms else None,
        "ground_tolerance_m": float(config["ground_tolerance_m"]),
        "penetration_tolerance_m": float(config["penetration_tolerance_m"]),
        "violations": violations,
        "pass": not violations,
    }


def _update_placements(placements: Sequence[dict], contents: List[dict], config: dict) -> None:
    """Write the settled numbers back into ``furniture_info["placements"]``."""
    by_name = {entry["name"]: entry for entry in contents}
    for index, placement in enumerate(placements):
        name = placement.get("name") or f"item_{index}"
        entry = by_name.get(name)
        if entry is None or entry["bbox_z_after"] is None:
            continue
        before, after = entry["bbox_z_before"], entry["bbox_z_after"]
        authored = placement.get("bbox_z")
        if authored is not None and not _close(authored, before, 0.002):
            placement["bbox_z_authored"] = authored
        placement["bbox_z"] = after
        if entry["mode"] == "rest":
            placement["z"] = after[0]
        placement["settle"] = {
            "mode": entry["mode"],
            "reason": entry["reason"],
            "bbox_z_before": before,
            "bbox_z_after": after,
            "moved_m": entry["moved_m"],
            "rotation_deg": entry["rotation_deg"],
            "drop_m": float(config["drop_m"]) if entry["mode"] == "rest" else 0.0,
            "floor_lift_m": round(float(entry["lift_m"]) - (float(config["drop_m"]) if entry["mode"] == "rest" else 0.0), 5),
            "separation_m": entry["separation_m"],
            "flat": entry["flat"],
            "objects": entry["objects"],
        }
        if "grounded" in entry:
            placement["settle"].update({
                "bottom_gap_m": entry["gap_m"],
                "grounded": entry["grounded"],
                "penetrating": entry["penetrating"],
            })
        if entry.get("support"):
            placement["settle"]["support"] = entry["support"]


def _close(a, b, tolerance: float) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= tolerance for x, y in zip(a, b))


def _sync_lamp_lights(spec, placements: Sequence[dict], lamp_lights: List, hulls: List[dict]) -> None:
    """Move a point light along with its lamp when settling moved the lamp.

    ``place_furniture`` appends one light per ceiling-mounted model and per
    ``builtin:table_lamp``, in placement order.  Both are normally *held*, so this
    is a guard for the odd floor-standing lamp rather than the normal path.
    """
    if not lamp_lights:
        return
    items = list(getattr(spec, "furniture", None) or [])
    by_name = {hull["name"]: hull for hull in hulls}
    lamp_index = 0
    for index, placement in enumerate(placements):
        if lamp_index >= len(lamp_lights):
            break
        item = items[index] if index < len(items) else None
        mount = getattr(item, "mount", None) or placement.get("mount") or "floor"
        if mount != "ceiling" and placement.get("asset") != "builtin:table_lamp":
            continue
        hull = by_name.get(placement.get("name") or f"item_{index}")
        light = lamp_lights[lamp_index]
        lamp_index += 1
        if hull is None or hull["mode"] != "rest":
            continue
        obj = hull["object"]
        delta = obj.matrix_world.translation - hull["matrix_before"][obj.name].translation
        lamp_lights[lamp_index - 1] = (
            float(light[0]) + float(delta.x), float(light[1]) + float(delta.y), float(light[2]) + float(delta.z),
        )


def _finish(report: dict, build_dir: Optional[str], started: float) -> dict:
    report.setdefault("seconds", round(time.time() - started, 2))
    if build_dir:
        path = os.path.join(build_dir, "settle_report.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        report["report_path"] = path
    return report


# --------------------------------------------------------------------------- #
# CLI: headless self-check on placed boxes
# --------------------------------------------------------------------------- #
def selfcheck(build_dir: Optional[str] = None, **overrides) -> dict:
    """Build a tiny scene of placed boxes and verify the settling contract.

    Covers a floating box (must land), a box sunk into the floor slab (must be
    lifted out, not fall through), an interpenetrating pair (must be separated
    without being launched), a stack (the upper box rests on the lower one), a
    thin rug-like pad with a box standing on it, a wall-flush box (must be pushed
    out of the wall), wall-mounted boxes (must not move at all) and a collection
    instance (the instancing path).  Returns a report with a ``pass`` flag.
    """
    bpy = _bpy()
    from blender_agent import shell as shell_mod
    from blender_agent import spec as spec_mod

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    shell_collection = shell_mod.ensure_collection(shell_mod.SHELL_COLLECTION)
    furniture = shell_mod.ensure_collection(FURNITURE_COLLECTION)
    shell_mod.add_prism("floor_selfcheck", [(0, 0), (5, 0), (5, 4), (0, 4)], -0.05, 0.0, shell_collection)
    shell_mod.add_prism("wall_east", [(4.94, 0), (5.06, 0), (5.06, 4), (4.94, 4)], 0.0, 2.7, shell_collection)

    boxes: Dict[str, object] = {}

    def box(name, x0, y0, x1, y1, z0, z1):
        boxes[name] = shell_mod.add_prism(name, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z0, z1, furniture)
        return boxes[name]

    box("drop_box", 0.50, 0.50, 1.00, 1.00, 0.080, 0.580)     # 8 cm above the floor
    box("sunk_box", 1.50, 0.50, 2.00, 1.00, -0.025, 0.475)    # 2.5 cm into the floor slab
    box("pair_a", 2.50, 0.50, 3.00, 1.00, 0.020, 0.520)       # 10 cm overlap with pair_b
    box("pair_b", 2.90, 0.50, 3.40, 1.00, 0.020, 0.520)
    box("stack_lower", 0.50, 2.50, 1.10, 3.10, 0.020, 0.420)
    box("upper_box", 0.60, 2.60, 1.00, 3.00, 0.460, 0.760)   # 4 cm above stack_lower: falls onto it
    box("rug_pad", 2.00, 1.80, 3.40, 3.20, 0.000, 0.012)      # rug-like slab
    box("pad_box", 2.40, 2.20, 2.80, 2.60, 0.020, 0.420)      # stands on the pad
    box("held_box", 3.50, 1.00, 3.90, 1.40, 1.200, 1.600)     # wall-hung (authored z)
    box("ceiling_box", 4.00, 1.00, 4.40, 1.40, 2.300, 2.600)  # mount: ceiling
    box("flush_box", 4.60, 2.00, 5.00, 2.40, 0.020, 0.420)    # 6 cm into the east wall

    library = bpy.data.collections.new("selfcheck_library")   # not linked: instance source only
    shell_mod.add_prism("selfcheck_src",
                        [(-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)], 0.0, 0.4, library)
    instance = bpy.data.objects.new("inst_box", None)
    instance.instance_type = "COLLECTION"
    instance.instance_collection = library
    instance.location = (4.30, 3.50, 0.05)                    # 5 cm above the floor
    furniture.objects.link(instance)

    furniture_spec = {
        "unit": "meter", "wall_height": 2.7, "wall_thickness": 0.12,
        "rooms": [{"id": "room", "type": "living_room", "polygon": [[0, 0], [5, 0], [5, 4], [0, 4]]}],
        "cameras": [{"name": "0000", "room": "room", "pos": [2.5, 2.0, 1.5]}],
        "furniture": [
            {"asset": "ph:selfcheck", "room": "room", "pos": [0.75, 0.75], "name": "drop_box"},
            {"asset": "ph:selfcheck", "room": "room", "pos": [1.75, 0.75], "name": "sunk_box"},
            {"asset": "ph:selfcheck", "room": "room", "pos": [2.75, 0.75], "name": "pair_a"},
            {"asset": "ph:selfcheck", "room": "room", "pos": [3.15, 0.75], "name": "pair_b"},
            {"asset": "ph:selfcheck", "room": "room", "pos": [0.80, 2.80], "name": "stack_lower"},
            {"asset": "ph:selfcheck", "room": "room", "pos": [0.80, 2.80], "name": "upper_box"},
            {"asset": "builtin:rug", "room": "room", "pos": [2.70, 2.50], "size": [1.4, 1.4], "name": "rug_pad"},
            {"asset": "ph:selfcheck", "room": "room", "pos": [2.60, 2.40], "name": "pad_box"},
            {"asset": "ph:selfcheck", "room": "room", "pos": [3.70, 1.20], "z": 1.20, "name": "held_box"},
            {"asset": "ph:selfcheck", "room": "room", "pos": [4.20, 1.20], "mount": "ceiling",
             "name": "ceiling_box"},
            {"asset": "ph:selfcheck", "room": "room", "pos": [4.80, 2.20], "name": "flush_box"},
            {"asset": "ph:selfcheck", "room": "room", "pos": [4.30, 3.50], "name": "inst_box"},
        ],
    }
    # the placements mirror what place_furniture records: ``z`` is the *authored*
    # mount height (0 for a piece that is meant to stand on the floor), while the
    # geometry may well start floating, sunk or overlapping
    spec = spec_mod.parse_spec(furniture_spec)
    placements = [
        {"name": "drop_box", "asset": "ph:selfcheck", "kind": "model", "instance": "drop_box",
         "pos": [0.75, 0.75], "rot_z_deg": 0.0, "z": 0.0, "mount": "floor", "bbox_z": [0.0, 0.5]},
        {"name": "sunk_box", "asset": "ph:selfcheck", "kind": "model", "instance": "sunk_box",
         "pos": [1.75, 0.75], "rot_z_deg": 0.0, "z": 0.0, "mount": "floor", "bbox_z": [0.0, 0.5]},
        {"name": "pair_a", "asset": "ph:selfcheck", "kind": "model", "instance": "pair_a",
         "pos": [2.75, 0.75], "rot_z_deg": 0.0, "z": 0.0, "mount": "floor", "bbox_z": [0.0, 0.5]},
        {"name": "pair_b", "asset": "ph:selfcheck", "kind": "model", "instance": "pair_b",
         "pos": [3.15, 0.75], "rot_z_deg": 0.0, "z": 0.0, "mount": "floor", "bbox_z": [0.0, 0.5]},
        {"name": "stack_lower", "asset": "ph:selfcheck", "kind": "model", "instance": "stack_lower",
         "pos": [0.80, 2.80], "rot_z_deg": 0.0, "z": 0.0, "mount": "floor", "bbox_z": [0.0, 0.4]},
        {"name": "upper_box", "asset": "ph:selfcheck", "kind": "model", "instance": "upper_box",
         "pos": [0.80, 2.80], "rot_z_deg": 0.0, "z": 0.0, "mount": "floor", "bbox_z": [0.0, 0.3]},
        {"name": "rug_pad", "asset": "builtin:rug", "kind": "builtin",
         "pos": [2.70, 2.50], "rot_z_deg": 0.0, "z": 0.0, "size_m": [1.4, 1.4, 0.012]},
        {"name": "pad_box", "asset": "ph:selfcheck", "kind": "model", "instance": "pad_box",
         "pos": [2.60, 2.40], "rot_z_deg": 0.0, "z": 0.0, "mount": "floor", "bbox_z": [0.0, 0.4]},
        {"name": "held_box", "asset": "ph:selfcheck", "kind": "model", "instance": "held_box",
         "pos": [3.70, 1.20], "rot_z_deg": 0.0, "z": 1.20, "mount": "floor", "bbox_z": [1.2, 1.6]},
        {"name": "ceiling_box", "asset": "ph:selfcheck", "kind": "model", "instance": "ceiling_box",
         "pos": [4.20, 1.20], "rot_z_deg": 0.0, "z": 2.30, "mount": "ceiling", "bbox_z": [2.3, 2.6]},
        {"name": "flush_box", "asset": "ph:selfcheck", "kind": "model", "instance": "flush_box",
         "pos": [4.80, 2.20], "rot_z_deg": 0.0, "z": 0.0, "mount": "floor", "bbox_z": [0.0, 0.4]},
        {"name": "inst_box", "asset": "ph:selfcheck", "kind": "model", "instance": "inst_box",
         "pos": [4.30, 3.50], "rot_z_deg": 0.0, "z": 0.0, "mount": "floor", "bbox_z": [0.0, 0.4]},
    ]
    report = ground_furniture(spec, {"placements": placements, "lamp_lights": []}, build_dir,
                              shell_info=None, **overrides)

    entries = {entry["name"]: entry for entry in report["objects"]}
    checks: List[dict] = []

    def check(name, passed, detail):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    def after(name):
        return entries[name]["bbox_z_after"]

    def extent(name, axis):
        obj = boxes[name]
        values = [(obj.matrix_world @ vertex.co)[axis] for vertex in obj.data.vertices]
        return min(values), max(values)

    grounding = ("drop_box", "sunk_box", "pair_a", "pair_b", "stack_lower", "rug_pad", "pad_box",
                 "flush_box", "inst_box")
    ground = report["checks"]["ground_tolerance_m"]
    for name in grounding:
        bottom = after(name)[0]
        check(f"{name} on the floor", -0.001 <= bottom <= ground,
              f"bottom {bottom * 1000:.1f} mm (allowed 0..{ground * 1000:.0f} mm)")
    check("nothing sank into the floor", all(after(name)[0] >= -0.001 for name in grounding),
          "min bottom "
          f"{min(after(name)[0] for name in grounding) * 1000:.1f} mm over {len(grounding)} boxes")
    check("drop_box fell to rest",
          entries["drop_box"]["bbox_z_before"][0] > 0.05 and after("drop_box")[0] <= ground,
          f"{entries['drop_box']['bbox_z_before'][0] * 1000:.0f} mm -> {after('drop_box')[0] * 1000:.1f} mm")
    check("sunk_box lifted out of the slab",
          entries["sunk_box"]["bbox_z_before"][0] < -0.001 and after("sunk_box")[0] >= -0.001,
          f"started {entries['sunk_box']['bbox_z_before'][0] * 1000:.1f} mm into the slab, now "
          f"{after('sunk_box')[0] * 1000:.1f} mm")
    gap = extent("pair_b", 0)[0] - extent("pair_a", 0)[1]
    check("overlapping pair separated", -0.001 <= gap <= 0.05,
          f"x gap {gap * 1000:.1f} mm, pair_a x {tuple(round(v, 3) for v in extent('pair_a', 0))}, "
          f"pair_b x {tuple(round(v, 3) for v in extent('pair_b', 0))}")
    stack_gap = after("upper_box")[0] - after("stack_lower")[1]
    check("upper_box rests on stack_lower", -0.001 <= stack_gap <= ground,
          f"gap {stack_gap * 1000:.1f} mm (upper bottom {after('upper_box')[0]:.3f} m, "
          f"lower top {after('stack_lower')[1]:.3f} m)")
    check("rug_pad stays flat and low",
          -0.001 <= after("rug_pad")[0] <= ground and after("rug_pad")[1] <= 0.03,
          f"z {after('rug_pad')[0] * 1000:.1f}..{after('rug_pad')[1] * 1000:.1f} mm")
    for name in ("held_box", "ceiling_box"):
        before, now = entries[name]["bbox_z_before"], after(name)
        check(f"{name} held in place",
              entries[name]["mode"] == "held" and abs(now[0] - before[0]) < 0.001,
              f"mode {entries[name]['mode']}, bottom {before[0]:.3f} -> {now[0]:.3f} m")
    max_x = extent("flush_box", 0)[1]
    check("flush_box pushed out of the wall", max_x <= 4.941,
          f"wall face at x=4.940, box reaches x={max_x:.4f}")
    check("instance rests and its empty moved",
          entries["inst_box"]["moved_m"] > 0.02 and after("inst_box")[0] <= ground,
          f"empty moved {entries['inst_box']['moved_m'] * 1000:.1f} mm, bottom "
          f"{after('inst_box')[0] * 1000:.1f} mm")
    check("everything came to rest", bool(report["simulation"]["at_rest"]),
          f"{report['simulation']['frames_used']} frames, peak motion "
          f"{report['simulation']['peak_motion_m']} m")
    check("settle checks pass", bool(report["checks"]["pass"]),
          "; ".join(report["checks"]["violations"]) or "no violations")

    result = {
        "kind": "blender_agent_settle_selfcheck",
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
        "report": report,
    }
    if build_dir:
        with open(os.path.join(build_dir, "settle_selfcheck.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    return result


def _print_selfcheck(result: dict) -> None:
    print("settle self-check (placed boxes):")
    for item in result["checks"]:
        print(f"  [{'PASS' if item['pass'] else 'FAIL'}] {item['check']:<35} {item['detail']}")
    print(f"  -> {'PASS' if result['pass'] else 'FAIL'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="settle.py", description=__doc__.splitlines()[0])
    parser.add_argument("--selfcheck", action="store_true",
                        help="build a box scene and verify the settling contract (the default action)")
    parser.add_argument("--json", help="directory to write settle_selfcheck.json into")
    parser.add_argument("--margin", type=float, help="collision margin (m): the resting gap")
    parser.add_argument("--drop", type=float, help="drop height (m)")
    parser.add_argument("--frames", type=int, help="simulation budget in frames")
    parser.add_argument("--substeps", type=int, help="Bullet substeps per frame")
    parser.add_argument("--solver-iterations", type=int, help="Bullet solver iterations")
    args = parser.parse_args(list(argv) if argv is not None else None)

    overrides = {key: value for key, value in (
        ("margin_m", args.margin), ("drop_m", args.drop), ("frames", args.frames),
        ("substeps", args.substeps), ("solver_iterations", args.solver_iterations),
    ) if value is not None}
    result = selfcheck(build_dir=args.json, **overrides)
    _print_selfcheck(result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(argv))
