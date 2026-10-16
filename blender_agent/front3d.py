"""3D-FRONT scene replay: parse the official scene JSON and rebuild it in Blender.

The dataset has not landed on this host yet (the official application is in
flight), so everything here is written against the published format and is
exercised end to end on ``tests/fixtures/front3d_mini``.  Drop the real folders
in and the same code path runs unchanged.

Format truth
------------
Read off the two authoritative parsers and then confirmed against a real scene
JSON (``3D-FRONT/9a3df633-792d-4e06-a138-ef4efa41959d.json``, 478 shell meshes,
36 furniture items, 9 rooms, ``version: 2.0``, pulled from the official
distribution by HTTP range request):

* official toolbox -- ``3D-FRONT-FUTURE/3D-FRONT-ToolBox``, ``scripts/scene.py``
  (JSON reader: ``furniture`` / ``material`` / ``mesh`` / ``scene.room``) and
  ``scripts/utils.py`` (``transform_v``, ``quaternion_to_matrix``,
  ``read_mesh_attr``); ``Room/process_json.py`` in the same repo reads the same
  keys, and so does the 2020 revision of ``scripts/json2obj.py`` (it applies
  ``v*scale -> R*yaw -> +pos`` and recovers the yaw as a rotation about **+Y**,
  independently confirming both the transform order and the up axis).  Both
  generations of the toolkit therefore agree with BlenderProc on this format;
  the older ``layouts`` / ``floors`` / ``furnitures`` document that circulates
  in some write-ups is *not* what the 3D-FRONT distribution ships as a scene
  JSON, and no revision of the toolkit's ``json2obj.py`` reads it.
* BlenderProc -- ``blenderproc/python/loader/Front3DLoader.py`` (mesh -> Blender
  mesh, jid -> ``raw_model.obj``, pos/rot/scale application, texture lookup)

1. **Frame**: the scene JSON is right-handed **Y-up**, unit **metres**, floor
   plane ``y = 0``.  Shell "Floor" meshes have ``y == 0.0`` exactly, walls span
   ``y in [0, 2.8]`` and ``extension.snapshots[].up`` is ``[0, 1, 0]``.
2. **Shell geometry is inline in the JSON**, not an OBJ: ``mesh[]`` entries carry
   ``xyz`` / ``normal`` / ``uv`` (flat, per-vertex) and ``faces`` (flat, one
   triangle per three indices).  ``json2obj.py`` only *exports* those vertices to
   OBJ; the shell "OBJ" in a converted tree is that export.  So the faithful
   replay imports the inline triangles and never extrudes anything.
3. **Mesh types** are free-form strings (``WallInner``, ``WallOuter``,
   ``Floor``, ``Ceiling``, ``SlabSide``, ``Baseboard``, ``Cornice``,
   ``Customized_wainscot``, ``Window``, ``Pocket``, ``""``, ...).  The toolbox
   folds them to ``ceil`` / ``floor`` / ``wall`` / ``others`` by substring.
4. **Room assembly**: ``scene.room[]`` has ``type`` / ``instanceid`` /
   ``children[]``; each child has ``ref`` + ``pos`` / ``rot`` / ``scale``.  A
   ``ref`` matching a ``mesh[].uid`` adds that mesh to the room -- measured on
   the real scene, **all 478 mesh children carry the identity transform**, so
   mesh vertices are already in scene coordinates (this parser still applies a
   non-identity child transform if one ever shows up).  A ``ref`` matching a
   ``furniture[].uid`` places one furniture instance.
5. **Furniture transform**: ``v_scene = R(q) @ (s * v_model) + pos`` -- scale
   element-wise first, rotation second, translation last (``transform_v`` in the
   official toolbox; BlenderProc composes the same order through the object
   matrix).  ``s`` is a **3-vector**: ``[1.499, 1.132, 1.0]`` occurs in real
   data, so a single uniform scale would be wrong.
6. **Quaternion order is ``(x, y, z, w)``**: ``quaternion_to_matrix`` reads
   ``args[3]`` as ``w``; identity in the data is ``[0, 0, 0, 1]``; and every
   rotation in the real scene is a pure yaw about +Y (``|x| = |z| = 0``,
   ``|y|`` up to 1).  Blender's ``mathutils.Quaternion`` takes ``(w, x, y, z)``,
   which is why the BlenderProc loader's ``mathutils.Quaternion(child["rot"])``
   must not be copied blindly.
7. **Furniture models** live at ``<3D-FUTURE-model>/<jid>/``: ``raw_model.obj``
   (+ ``model.mtl`` + ``texture.png``).  BlenderProc reads ``raw_model.obj``;
   the toolbox reads ``raw_model_tri.obj`` (its own triangulated export).  Models
   are authored in the same Y-up frame as the scene, so the instance transform
   applies to raw model vertices with no extra conversion.
8. **Model origin is the bottom-centre of the bounding box**: the geometry spans
   ``y in [0, height]`` and the instance ``pos`` is the lowest point.  Measured
   on the real scene: every floor-standing instance has ``pos.y == 0.000`` and
   ``pos.y + bbox.y ==`` the model's top; pendant lamps, whose geometry also
   spans ``[0, h]``, hang with their top at the ceiling.  This is the same
   convention as this repo's Poly Haven assets (``assets.py``: bottom-centre
   origin, ``z = 0`` stands on the floor) with Y as up instead of Z.
9. **Room textures**: ``material[]`` entries are shared by uid across meshes and
   carry ``texture`` (a URL; the second-to-last path segment is the folder name
   under ``3D-FRONT-textures``) and/or a flat ``color`` (0-255, sRGB-ish).
   ``json2obj.py`` downloads ``<texture_root>/<hash>/texture.png`` lazily.
10. **Lights**: ``lights[]`` are VRay light definitions (``src_position``,
    ``multiplier``, ...).  They are parsed and reported but deliberately *not*
    imported -- this repo's stage lights the scene physically (Nishita sky +
    sun + the enclosing sky dome), and a VRay "multiplier" has no meaning in
    Cycles.
11. **Official viewpoints** are in ``extension.snapshots[]`` (``pos`` / ``target``
    / ``up`` in the front frame -- ``up`` is ``[0, 1, 0]``, another confirmation
    of the up axis).  ``--camera-source snapshots`` uses those positions as the
    replay's cameras; the yaw is fixed to 0 because the PanoWorld pose contract
    is ``spec.R0`` (yaw 0), so a snapshot's looking direction is reported as
    ``source_snapshot`` metadata but never rotated in.

Coordinate conversion (front frame -> Blender/repo frame)
--------------------------------------------------------
The repo renders in Blender's Z-up frame (``spec.py``: ``R0``, floors at
``z = 0``).  Front -> Blender is a **proper rotation** ``M = Rx(+90 deg)``::

    M = [[1, 0, 0, 0],
         [0, 0, -1, 0],      (x, y, z)_front -> (x, -z, y)_blender
         [0, 1, 0, 0],
         [0, 0, 0, 1]]

* BlenderProc instead *swaps* y and z (``Vector(child["pos"]).xzy``), which is a
  mirror (determinant -1) and flips triangle winding; a rotation does not, so a
  normal computed by Blender still points the same way relative to the surface.
* ``M`` maps front +Y (up) to Blender +Z (up) and the model's front axis +Z to
  Blender -Y -- exactly the "front faces -Y" convention ``assets.py`` uses, so a
  dataset yaw becomes this repo's ``rot_z_deg`` unchanged (counter-clockwise
  about +Z, 0 = facing -Y).
* The world matrix of one instance is therefore
  ``M @ T(pos) @ R(q) @ S(scale)`` applied to *raw* front-frame model vertices;
  the OBJ import is done with ``forward_axis="Y", up_axis="Z"`` precisely so the
  importer adds no conversion of its own.

Everything above is asserted in ``tests/test_front3d_math.py`` (hand-computed
poses, < 1e-6) and end to end in ``tests/test_front3d_replay.sh``.

Host side (no ``bpy``, run with the repo python)::

    python blender_agent/front3d.py --scene 3D-FRONT/<uid>.json \\
        --model-root 3D-FUTURE-model --dry-run
    python blender_agent/front3d.py --scene ... --model-root ... \\
        --spec-out /tmp/front3d_scene_spec.json

Blender side: ``run_build.py --front3d-scene ... --front3d-model-root ...``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# front frame (Y-up) -> repo/Blender frame (Z-up), see the module docstring
FRONT_TO_BLENDER = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

DEFAULT_EYE_HEIGHT = 1.5
DEFAULT_WALL_THICKNESS = 0.12
DEFAULT_WALL_HEIGHT = 2.8
FURNITURE_SCHEME = "front3d"

# room function inference: category token -> function (longest match wins)
_FUNCTION_KEYWORDS = (
    ("bed", "bedroom"),
    ("kids bed", "bedroom"),
    ("bunk bed", "bedroom"),
    ("wardrobe", "bedroom"),
    ("nightstand", "bedroom"),
    ("night table", "bedroom"),
    ("dressing table", "bedroom"),
    ("sofa", "living"),
    ("coffee table", "living"),
    ("tv", "living"),
    ("media unit", "living"),
    ("armchair", "living"),
    ("dining table", "dining"),
    ("dining chair", "dining"),
    ("bar ", "dining"),
    ("wine cabinet", "dining"),
    ("kitchen", "kitchen"),
    ("gas stove", "kitchen"),
    ("sink", "kitchen"),
    ("fridge", "kitchen"),
    ("microwave", "kitchen"),
    ("toilet", "bathroom"),
    ("bath", "bathroom"),
    ("shower", "bathroom"),
    ("washbasin", "bathroom"),
    ("desk", "study"),
    ("bookshelf", "study"),
    ("bookcase", "study"),
    ("study", "study"),
    ("crib", "nursery"),
)

_ROOM_TYPE_FUNCTION = {
    "livingroom": "living",
    "livingdiningroom": "living_dining",
    "diningroom": "dining",
    "bedroom": "bedroom",
    "masterbedroom": "bedroom",
    "secondbedroom": "bedroom",
    "kidsroom": "bedroom",
    "kitchen": "kitchen",
    "bathroom": "bathroom",
    "masterbathroom": "bathroom",
    "studyingroom": "study",
    "studyroom": "study",
    "officeroom": "study",
    "corridor": "circulation",
    "balcony": "outdoor",
    "otherroom": "other",
}


# --------------------------------------------------------------------------- #
# transform math (pure stdlib: shared by the host CLI, the tests and Blender)
# --------------------------------------------------------------------------- #
def identity4():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def mat4_mul(a, b):
    """4x4 matrix product ``a @ b`` (row-major lists)."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def mat4_apply(matrix, point):
    """Transform a 3-vector by a 4x4 matrix (w = 1, no projective divide)."""
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    return [
        matrix[i][0] * x + matrix[i][1] * y + matrix[i][2] * z + matrix[i][3]
        for i in range(3)
    ]


def mat4_inverse_rotation(matrix):
    """Inverse of a rigid transform (rotation + translation only)."""
    rot = [[matrix[i][j] for j in range(3)] for i in range(3)]
    trans = [matrix[i][3] for i in range(3)]
    inv_rot = [[rot[j][i] for j in range(3)] for i in range(3)]
    inv_trans = [-sum(inv_rot[i][k] * trans[k] for k in range(3)) for i in range(3)]
    return [
        [inv_rot[i][0], inv_rot[i][1], inv_rot[i][2], inv_trans[i]]
        for i in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def quaternion_xyzw_to_matrix(quat, normalize: bool = True):
    """Rotation matrix of an ``(x, y, z, w)`` quaternion.

    Same arithmetic as ``utils.quaternion_to_matrix`` in the official toolbox
    (``args[3]`` is ``w``); the quaternion is rescaled first when it is not
    already unit length (the toolbox does not, real data is unit to ~1e-7).
    """
    if len(quat) != 4:
        raise ValueError(f"quaternion must have 4 components, got {quat!r}")
    x, y, z, w = (float(v) for v in quat)
    if normalize:
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            raise ValueError("degenerate quaternion (zero norm)")
        if abs(norm - 1.0) > 1e-9:
            x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def scene_instance_matrix(pos, rot, scale, frame=True):
    """``M @ T(pos) @ R(rot) @ S(scale)`` for one 3D-FRONT placement.

    ``frame=False`` keeps the result in the dataset's own Y-up frame (used by
    the tests and by the spec bridge); ``frame=True`` (default) composes the
    front -> Blender conversion on the left, i.e. the matrix to hand to Blender
    for *raw* front-frame vertices.
    """
    if isinstance(scale, (int, float)):
        scale = [float(scale)] * 3
    if len(scale) != 3:
        raise ValueError(f"scale must be a 3-vector or a scalar, got {scale!r}")
    rot_matrix = quaternion_xyzw_to_matrix(rot)
    # T @ R @ S, so a point is scaled, then rotated, then translated
    matrix = [
        [rot_matrix[i][j] * float(scale[j]) for j in range(3)] + [float(pos[i])]
        for i in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]
    return mat4_mul(FRONT_TO_BLENDER, matrix) if frame else matrix


def front_to_blender_point(point):
    return [float(point[0]), -float(point[2]), float(point[1])]


def blender_to_front_point(point):
    return [float(point[0]), float(point[2]), -float(point[1])]


def yaw_deg_from_quaternion(quat):
    """Yaw about the dataset's +Y (up) axis, in degrees.

    Returns ``(yaw_deg, off_axis)`` where ``off_axis`` is the largest deviation
    from a pure yaw (``max(|R[1][0]|, |R[1][2]|)``); every rotation in the real
    dataset yields ``off_axis == 0`` to machine precision.
    """
    rot = quaternion_xyzw_to_matrix(quat)
    yaw = math.degrees(math.atan2(rot[0][2], rot[0][0]))
    off_axis = max(abs(rot[1][0]), abs(rot[1][2]), abs(rot[1][1] - 1.0))
    return yaw, off_axis


def bbox_of_points(points):
    if not points:
        return None
    return [
        [min(p[i] for p in points) for i in range(3)],
        [max(p[i] for p in points) for i in range(3)],
    ]


def transform_bbox(matrix, bbox):
    """World AABB of an axis-aligned local box under an affine transform."""
    lo, hi = bbox
    corners = [
        [lo[0] if (bits & 1) else hi[0], lo[1] if (bits & 2) else hi[1], lo[2] if (bits & 4) else hi[2]]
        for bits in range(8)
    ]
    return bbox_of_points([mat4_apply(matrix, corner) for corner in corners])


# --------------------------------------------------------------------------- #
# geometry helpers used to recover room polygons from the shell meshes
# --------------------------------------------------------------------------- #
def polygon_signed_area(polygon) -> float:
    area = 0.0
    count = len(polygon)
    for i in range(count):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % count]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def point_in_polygon(point, polygon) -> bool:
    """Even-odd rule, matching ``spec.point_in_polygon``."""
    x, y = float(point[0]), float(point[1])
    inside = False
    count = len(polygon)
    for i in range(count):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % count]
        if (y0 > y) != (y1 > y):
            t = (y - y0) / (y1 - y0)
            if x < x0 + t * (x1 - x0):
                inside = not inside
    return inside


def distance_to_polygon(point, polygon) -> float:
    """Distance from a point to the boundary of a polygon (0 on the boundary)."""
    x, y = float(point[0]), float(point[1])
    best = float("inf")
    count = len(polygon)
    for i in range(count):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % count]
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq < 1e-12 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
        best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best


def convex_hull(points):
    """Andrew's monotone chain; returns a counter-clockwise hull (no repeats)."""
    unique = sorted({(round(float(p[0]), 9), round(float(p[1]), 9)) for p in points})
    if len(unique) <= 2:
        return [list(p) for p in unique]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return [list(p) for p in hull] if len(hull) >= 3 else [list(p) for p in unique]


def boundary_loops(triangles, tolerance: float = 1e-6):
    """Closed boundary loops of a triangle soup, as vertex-index loops.

    Used to recover a room's floor outline (3D-FRONT floor meshes are flat
    triangulated polygons): shared edges cancel, the rest form the outline.
    """
    edges = {}
    for triangle in triangles:
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1
    adjacency = {}
    for (a, b), count in edges.items():
        if count == 1:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
    loops = []
    visited = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        current, previous = start, None
        closed = False
        while True:
            options = [n for n in adjacency.get(current, []) if n != previous]
            if start in options:                # back at the first vertex: done
                closed = len(loop) >= 3
                break
            following = next((n for n in options if n not in visited), None)
            if following is None:               # dangling / non-manifold vertex
                break
            loop.append(following)
            visited.add(following)
            previous, current = current, following
        if closed:
            loops.append(loop)
    return loops


def welded_mesh(pieces, tolerance: float = 6):
    """Weld several 2D triangle pieces into one index space.

    A 3D-FRONT room floor can be split over several ``Floor`` meshes; welding by
    rounded coordinates (``tolerance`` decimal digits) makes the shared edges
    cancel in :func:`boundary_loops`, which yields the outline of the *union*.
    """
    index = {}
    weld = []
    triangles = []
    for points, piece_triangles in pieces:
        local = []
        for point in points:
            key = (round(float(point[0]), tolerance), round(float(point[1]), tolerance))
            if key not in index:
                index[key] = len(weld)
                weld.append([key[0], key[1]])
            local.append(index[key])
        for a, b, c in piece_triangles:
            triangles.append((local[a], local[b], local[c]))
    return weld, triangles


def outline_of_pieces(pieces, tolerance: float = 6):
    """``(largest_outline, total_area, loop_areas)`` for a set of 2D pieces."""
    points, triangles = welded_mesh(pieces, tolerance)
    found = []
    for loop in boundary_loops(triangles):
        polygon = [points[i] for i in loop]
        found.append((abs(polygon_signed_area(polygon)), polygon))
    found.sort(key=lambda item: -item[0])
    total = sum(area for area, _ in found)
    return (found[0][1] if found else []), total, [area for area, _ in found]


# --------------------------------------------------------------------------- #
# parsed representation
# --------------------------------------------------------------------------- #
class Front3DScene:
    """Everything ``parse_scene`` recovers from one 3D-FRONT scene JSON."""

    def __init__(self, uid, version, path, rooms, meshes, materials, furniture, instances, lights, warnings):
        self.uid = uid
        self.version = version
        self.path = path
        self.rooms = rooms                    # list[Room]
        self.meshes = meshes                  # {uid: Mesh}
        self.materials = materials            # {uid: Material}
        self.furniture = furniture            # {uid: Furniture}
        self.instances = instances            # list[Instance]
        self.lights = lights                  # list[dict]
        self.warnings = warnings              # list[str]
        self.snapshots = []                   # extension.snapshots (official views)
        self.categories = {}                  # category -> count
        self.room_functions = {}              # room instanceid -> function
        self.category_functions = {}          # category -> dominant room function

    # -- derived facts ----------------------------------------------------- #
    def room_by_id(self, room_id):
        for room in self.rooms:
            if room.instanceid == room_id:
                return room
        return None

    def in_room(self, point_xy):
        """Smallest room containing a (x, y) point, else the nearest one.

        Smallest, not first: 3D-FRONT catch-all rooms (``OtherRoom``) can cover
        the whole house, and a point inside a bedroom is inside both.
        """
        containing = [
            room for room in self.rooms
            if room.polygon and point_in_polygon(point_xy, room.polygon)
        ]
        if containing:
            return min(containing, key=lambda room: room.area)
        best, best_distance = None, float("inf")
        for room in self.rooms:
            if not room.polygon:
                continue
            distance = distance_to_polygon(point_xy, room.polygon)
            if distance < best_distance:
                best, best_distance = room, distance
        return best

    def instance_records(self, include_invalid: bool = False):
        records = []
        for instance in self.instances:
            if instance.valid or include_invalid:
                records.append(instance)
        return records


class Front3DRoom:
    def __init__(self, type, instanceid, size, pos, rot, scale, children):
        self.type = type
        self.instanceid = instanceid
        self.size = size
        self.pos = pos
        self.rot = rot
        self.scale = scale
        self.children = children
        self.mesh_uids = []
        self.furniture_uids = []
        self.polygon = []                 # repo-frame XY outline (may be empty)
        self.polygon_source = None
        self.centroid = None
        self.camera_point = None
        self.function = "other"
        self.function_votes = {}
        self.catch_all = False            # mostly covered by smaller rooms
        self.overlap_ratio = None         # fraction of this outline inside another room
        self.outline_area = None          # area of the floor outline before simplification

    @property
    def area(self) -> float:
        return abs(polygon_signed_area(self.polygon)) if len(self.polygon) >= 3 else 0.0

    def categories(self, scene) -> dict:
        counts = {}
        for uid in self.furniture_uids:
            item = scene.furniture.get(uid)
            if item is None or not item.valid:
                continue
            name = item.category_key or "other"
            counts[name] = counts.get(name, 0) + 1
        return counts


class Front3DMaterial:
    def __init__(self, uid, jid, texture, normaltexture, color, uv_transform):
        self.uid = uid
        self.jid = jid
        self.texture = texture
        self.normaltexture = normaltexture
        self.color = color                 # [r, g, b, a] 0-255
        self.uv_transform = uv_transform   # 3x3 or None

    @property
    def texture_ids(self):
        """Candidate ``3D-FRONT-textures`` folder names, best first.

        Two revisions of the dataset disagree on where the texture folder name
        lives, so both rules are honoured: the official toolbox's 2022
        ``json2obj.py`` uses ``material.jid`` as the folder, BlenderProc instead
        takes the second-to-last segment of the ``texture`` URL (the hash) and
        downloads it when missing.
        """
        ids = []
        if self.jid:
            ids.append(self.jid)
        if self.texture:
            parts = str(self.texture).rstrip("/").split("/")
            if len(parts) >= 2 and parts[-2] and parts[-2] not in ids:
                ids.append(parts[-2])
        return ids

    @property
    def texture_id(self):
        ids = self.texture_ids
        return ids[0] if ids else None

    @property
    def rgb(self):
        return [float(max(0, min(255, c))) / 255.0 for c in self.color[:3]]

    @property
    def is_white(self) -> bool:
        return all(abs(c - 1.0) < 1e-3 for c in self.rgb)


class Front3DMesh:
    def __init__(self, uid, type, constructid, vertices, normals, uvs, faces, material_uid):
        self.uid = uid
        self.type = type
        self.constructid = constructid
        self.vertices = vertices           # flat [x, y, z, ...] in the front frame
        self.normals = normals
        self.uvs = uvs
        self.faces = faces                 # flat triangle indices
        self.material_uid = material_uid
        self.child_transform = None        # set when the room child is not identity

    @property
    def vertex_count(self) -> int:
        return len(self.vertices) // 3

    @property
    def triangle_count(self) -> int:
        return len(self.faces) // 3

    @property
    def kind(self) -> str:
        """``floor`` / ``ceiling`` / ``wall`` / ``other`` (the toolbox's rule)."""
        lowered = self.type.lower()
        if "ceil" in lowered:
            return "ceiling"
        if "floor" in lowered:
            return "floor"
        if "wall" in lowered:
            return "wall"
        return "other"

    def points(self):
        return [
            (self.vertices[i], self.vertices[i + 1], self.vertices[i + 2])
            for i in range(0, len(self.vertices) - 2, 3)
        ]


class Front3DFurniture:
    def __init__(self, uid, jid, title, valid, category, super_category, size, bbox, type):
        self.uid = uid
        self.jid = jid
        self.title = title
        self.valid = valid
        self.category = category
        self.super_category = super_category
        self.size = size
        self.bbox = bbox
        self.type = type

    @property
    def category_key(self):
        """Case-folded category, so ``Bed`` and ``bed`` count as one label."""
        return self.category.lower() if self.category else None

    @property
    def subcategory(self):
        return self.title.split("/", 1)[1] if "/" in self.title else None


class Front3DInstance:
    def __init__(self, room_id, furniture, pos, rot, scale):
        self.room_id = room_id
        self.furniture = furniture
        self.pos = pos
        self.rot = rot
        self.scale = scale
        self.yaw_deg = 0.0
        self.off_axis = 0.0

    @property
    def valid(self) -> bool:
        return self.furniture.valid

    @property
    def uid(self) -> str:
        return self.furniture.uid

    @property
    def jid(self) -> str:
        return self.furniture.jid

    @property
    def spec_pos(self):
        """(x, y) in the repo frame (Blender Z-up) and the height as ``z``."""
        x, y, z = (float(v) for v in self.pos)
        return [x, -z], y

    def matrix(self, frame: bool = True):
        return scene_instance_matrix(self.pos, self.rot, self.scale, frame=frame)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def load_model_index(model_root):
    """``3D-FUTURE-model/model_info.json`` keyed by ``model_id`` (optional)."""
    if not model_root:
        return {}
    path = os.path.join(model_root, "model_info.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            entries = json.load(handle)
    except (OSError, ValueError) as exc:  # a broken index must not kill the replay
        return {"__error__": str(exc)}
    index = {}
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and entry.get("model_id"):
            index[entry["model_id"]] = entry
    return index


def _category_from_title(title):
    if not title:
        return None
    head = title.split("/", 1)[0].strip()
    return head or None


def _room_type_function(room_type):
    key = "".join(ch for ch in (room_type or "").lower() if ch.isalnum() or ch == "-")
    key = key.split("-")[0]
    return _ROOM_TYPE_FUNCTION.get(key, "other")


def _category_function(category):
    if not category:
        return None
    lowered = category.lower()
    best = None
    for token, function in _FUNCTION_KEYWORDS:
        if token in lowered and (best is None or len(token) > len(best[0])):
            best = (token, function)
    return best[1] if best else None


def parse_scene(json_path, model_root=None, strict=False):
    """Parse a 3D-FRONT scene JSON into a :class:`Front3DScene`.

    ``model_root`` is optional: when it holds ``model_info.json`` the real
    ``category`` / ``super-category`` of every model is used, otherwise the
    category falls back to the first path segment of the furniture ``title``
    (the same rule BlenderProc applies).
    """
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "scene" not in data:
        raise ValueError(f"{json_path}: no 'scene' block -- not a 3D-FRONT scene JSON")

    warnings = []
    model_index = load_model_index(model_root)
    if "__error__" in model_index:
        warnings.append(f"model_info.json could not be read ({model_index['__error__']})")
        model_index = {}

    materials = {}
    for raw in data.get("material", []) or []:
        uid = raw.get("uid")
        if not uid:
            continue
        uv_transform = raw.get("UVTransform")
        if uv_transform and len(uv_transform) == 9:
            uv_transform = [uv_transform[i * 3:i * 3 + 3] for i in range(3)]
        else:
            uv_transform = None
        materials[uid] = Front3DMaterial(
            uid=uid,
            jid=str(raw.get("jid", "") or ""),
            texture=str(raw.get("texture", "") or ""),
            normaltexture=str(raw.get("normaltexture", "") or ""),
            color=[float(c) for c in (raw.get("color") or [255, 255, 255, 255])[:4]] or [1.0, 1.0, 1.0, 1.0],
            uv_transform=uv_transform,
        )

    meshes = {}
    for raw in data.get("mesh", []) or []:
        uid = raw.get("uid")
        if not uid:
            continue
        vertices = [float(v) for v in (raw.get("xyz") or []) if v is not None]
        if len(vertices) % 3:
            warnings.append(f"mesh {uid}: xyz length {len(vertices)} is not a multiple of 3, truncated")
            vertices = vertices[: len(vertices) // 3 * 3]
        uvs = raw.get("uv")
        if uvs is not None:
            uvs = [float(v) for v in uvs if v is not None]
            if len(uvs) != 2 * (len(vertices) // 3):
                warnings.append(
                    f"mesh {uid}: uv has {len(uvs)} values for {len(vertices) // 3} vertices, ignoring them"
                )
                uvs = None
        faces = [int(f) for f in (raw.get("faces") or [])]
        if len(faces) % 3:
            warnings.append(f"mesh {uid}: faces length {len(faces)} is not a multiple of 3, truncated")
            faces = faces[: len(faces) // 3 * 3]
        normals = raw.get("normal")
        if normals is not None:
            normals = [float(v) for v in normals if v is not None]
            if len(normals) != len(vertices):
                normals = None
        mesh = Front3DMesh(
            uid=uid,
            type=str(raw.get("type", "") or ""),
            constructid=str(raw.get("constructid", "") or ""),
            vertices=vertices,
            normals=normals,
            uvs=uvs,
            faces=faces,
            material_uid=raw.get("material"),
        )
        if mesh.material_uid is not None and mesh.material_uid not in materials:
            warnings.append(f"mesh {uid}: material {mesh.material_uid} is not defined, using the fallback")
            mesh.material_uid = None
        meshes[uid] = mesh

    furniture = {}
    for raw in data.get("furniture", []) or []:
        uid = raw.get("uid")
        if not uid:
            continue
        jid = str(raw.get("jid", "") or "")
        info = model_index.get(jid, {}) if jid else {}
        title = str(raw.get("title", "") or "")
        category = raw.get("category") or info.get("category") or _category_from_title(title)
        furniture[uid] = Front3DFurniture(
            uid=uid,
            jid=jid,
            title=title,
            valid=bool(raw.get("valid", True)),
            category=str(category) if category else None,
            super_category=info.get("super-category") or info.get("super_category"),
            size=[float(v) for v in raw.get("size", [])] if raw.get("size") else None,
            bbox=raw.get("bbox"),
            type=str(raw.get("type", "") or ""),
        )

    rooms = []
    instances = []
    for raw in (data.get("scene", {}) or {}).get("room", []) or []:
        room = Front3DRoom(
            type=str(raw.get("type", "") or ""),
            instanceid=str(raw.get("instanceid", "") or ""),
            size=raw.get("size"),
            pos=raw.get("pos"),
            rot=raw.get("rot"),
            scale=raw.get("scale"),
            children=list(raw.get("children", []) or []),
        )
        unmatched = []
        for child in room.children:
            ref = child.get("ref")
            if ref in meshes:
                room.mesh_uids.append(ref)
                transform = (
                    [float(v) for v in (child.get("pos") or [0.0, 0.0, 0.0])],
                    [float(v) for v in (child.get("rot") or [0.0, 0.0, 0.0, 1.0])],
                    [float(v) for v in (child.get("scale") or [1.0, 1.0, 1.0])],
                )
                if transform != ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]):
                    warnings.append(
                        f"room {room.instanceid}: mesh child {ref} carries a non-identity transform "
                        f"{transform} -- applied at import time"
                    )
                    meshes[ref].child_transform = transform
            elif ref in furniture:
                room.furniture_uids.append(ref)
                instance = Front3DInstance(
                    room_id=room.instanceid,
                    furniture=furniture[ref],
                    pos=[float(v) for v in child["pos"]],
                    rot=[float(v) for v in child["rot"]],
                    scale=[float(v) for v in child["scale"]],
                )
                instance.yaw_deg, instance.off_axis = yaw_deg_from_quaternion(instance.rot)
                if instance.off_axis > 1e-6:
                    warnings.append(
                        f"furniture {ref} ({furniture[ref].title}): rotation is not a pure yaw about the "
                        f"dataset up axis (off-axis {instance.off_axis:.3g}); the spec bridge keeps the "
                        f"quaternion but the Blender placement uses the yaw"
                    )
                instances.append(instance)
            else:
                unmatched.append(ref)
        if unmatched:
            sample = ", ".join(str(r) for r in unmatched[:3])
            warnings.append(
                f"room {room.instanceid}: {len(unmatched)} child refs match no entry in furniture[] or "
                f"mesh[] ({sample}{', ...' if len(unmatched) > 3 else ''}) -- the dataset's furniture list "
                f"is partial, BlenderProc drops these too"
            )
        rooms.append(room)

    scene = Front3DScene(
        uid=str(data.get("uid", "") or os.path.splitext(os.path.basename(json_path))[0]),
        version=data.get("version"),
        path=os.path.abspath(json_path),
        rooms=rooms,
        meshes=meshes,
        materials=materials,
        furniture=furniture,
        instances=instances,
        lights=list(data.get("lights", []) or []),
        warnings=warnings,
    )
    scene.snapshots = list(((data.get("extension") or {}).get("snapshots") or []))
    resolve_room_geometry(scene, strict=strict)
    infer_room_functions(scene)
    return scene


def resolve_room_geometry(scene, strict=False):
    """Fill in each room's repo-frame outline, centroid and camera point.

    Preferred source is the outline of the room's ``Floor`` meshes (the outline
    of their union, see :func:`outline_of_pieces`) -- on the real scene this
    reproduces the dataset's own ``room.size`` area to within 1 %.  A room with
    no ``Floor`` entry (3D-FRONT has catch-all rooms like ``OtherRoom``) falls
    back to the convex hull of its floor-level geometry, then of everything.
    """
    for room in scene.rooms:
        points = []            # repo-frame XY of everything in the room
        low_points = []        # repo-frame XY of its floor-level geometry
        floor_triangles = []
        fronts = [
            point[1]
            for uid in room.mesh_uids
            for point in scene.meshes[uid].points()
        ]
        floor_level = min(fronts) + 0.3 if fronts else 0.0
        for uid in room.mesh_uids:
            mesh = scene.meshes[uid]
            if mesh.kind == "floor":
                floor_triangles.append(mesh)
            for point in mesh.points():
                converted = front_to_blender_point(point)
                points.append((converted[0], converted[1]))
                if point[1] <= floor_level:
                    low_points.append((converted[0], converted[1]))

        polygon, source = [], None
        if floor_triangles:
            pieces = []
            for mesh in floor_triangles:
                transform = mesh.child_transform
                front_points = mesh.points()
                if transform is not None:
                    matrix = scene_instance_matrix(*transform)
                    front_points = [mat4_apply(matrix, point) for point in front_points]
                flat = [
                    (front_to_blender_point(point)[0], front_to_blender_point(point)[1])
                    for point in front_points
                ]
                triangles = [
                    (mesh.faces[i], mesh.faces[i + 1], mesh.faces[i + 2])
                    for i in range(0, len(mesh.faces), 3)
                ]
                pieces.append((flat, triangles))
            polygon, outline_area, loop_areas = outline_of_pieces(pieces)
            if len(polygon) >= 3:
                source = "floor_outline"
                if len(loop_areas) > 1:
                    scene.warnings.append(
                        f"room {room.instanceid}: {len(loop_areas)} floor loops "
                        f"({', '.join(f'{a:.2f}' for a in loop_areas[:4])} m2), kept the largest"
                    )
                # the dataset's own `size` is an independent area measurement, so
                # it is a real cross-check of the recovered outline
                if isinstance(room.size, (int, float)) and room.size > 0:
                    if abs(outline_area - float(room.size)) > max(0.5, 0.05 * float(room.size)):
                        scene.warnings.append(
                            f"room {room.instanceid}: recovered floor outline {outline_area:.2f} m2 vs the "
                            f"dataset's own size {float(room.size):.2f} m2"
                        )
                room.outline_area = outline_area
        if len(polygon) < 3 and len(low_points) >= 3:
            polygon = convex_hull(low_points)
            source = "floor_level_hull"
            scene.warnings.append(
                f"room {room.instanceid}: no Floor mesh, outline taken from the convex hull of its "
                f"{len(low_points)} floor-level vertices"
            )
        if len(polygon) < 3 and points:
            polygon = convex_hull(points)
            source = "mesh_hull"
            scene.warnings.append(
                f"room {room.instanceid}: no floor mesh outline, falling back to the convex hull of its "
                f"{len(points)} mesh vertices (the room has no Floor entry)"
            )
        if len(polygon) < 3:
            source = "none"
            scene.warnings.append(f"room {room.instanceid}: no geometry at all, room dropped from the spec")

        if polygon and polygon_signed_area(polygon) < 0.0:
            polygon.reverse()          # spec.py wants counter-clockwise polygons
        room.polygon = polygon
        room.polygon_source = source
        if polygon:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            room.centroid = [(min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0]
            room.camera_point = representative_point(polygon)

    # second pass: a room whose outline is mostly covered by *smaller* rooms is a
    # catch-all ("OtherRoom" style) -- real geometry, but not a viewpoint.  Only
    # smaller rooms count as covering, otherwise a catch-all that spans the whole
    # house would mark every real room as covered.
    for room in scene.rooms:
        if not room.polygon:
            continue
        others = [other.polygon for other in scene.rooms
                  if other is not room and other.polygon and other.area < room.area]
        if not others:
            room.overlap_ratio = 0.0
            continue
        samples = 12
        xs = [p[0] for p in room.polygon]
        ys = [p[1] for p in room.polygon]
        inside = total = 0
        for i in range(samples + 1):
            for j in range(samples + 1):
                x = min(xs) + (max(xs) - min(xs)) * i / samples
                y = min(ys) + (max(ys) - min(ys)) * j / samples
                if not point_in_polygon((x, y), room.polygon):
                    continue
                total += 1
                if any(point_in_polygon((x, y), other) for other in others):
                    inside += 1
        room.overlap_ratio = (inside / total) if total else 0.0
        if room.overlap_ratio > 0.6:
            room.catch_all = True
            scene.warnings.append(
                f"room {room.instanceid}: {100 * room.overlap_ratio:.0f}% of its outline is covered by "
                f"smaller rooms -- treated as a catch-all room, no camera placed here"
            )
    return scene


def representative_point(polygon, samples: int = 24):
    """A point inside the polygon, as central as a coarse grid can tell.

    The area centroid of an L-shaped room can fall outside it (and a camera
    outside its own room fails ``spec.validate_spec``), so the centroid is only
    used when it is inside; otherwise the grid point with the largest clearance
    from the boundary wins.
    """
    if len(polygon) < 3:
        return None
    area = polygon_signed_area(polygon)
    if abs(area) < 1e-12:
        return None
    cx = cy = 0.0
    count = len(polygon)
    for i in range(count):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % count]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    if point_in_polygon((cx, cy), polygon):
        return [cx, cy]

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    best, best_clearance = None, -1.0
    for i in range(samples + 1):
        for j in range(samples + 1):
            x = min(xs) + (max(xs) - min(xs)) * i / samples
            y = min(ys) + (max(ys) - min(ys)) * j / samples
            if not point_in_polygon((x, y), polygon):
                continue
            clearance = distance_to_polygon((x, y), polygon)
            if clearance > best_clearance:
                best, best_clearance = [x, y], clearance
    return best


def infer_room_functions(scene):
    """Fill ``room.function`` plus the category -> function tables.

    Two signals are combined, and both are reported: the furniture categories
    (which is what makes the table useful as a replacement for the VLM-guessed
    ``objects.json``) and the dataset's own ``room.type``.  A furniture vote
    only overrides the room type when at least two items agree, because single
    furnishing accidents are common (a lone desk in a living room).
    """
    tally = {}
    for room in scene.rooms:
        counts = room.categories(scene)
        votes = {}
        for category, count in counts.items():
            function = _category_function(category)
            if function:
                votes[function] = votes.get(function, 0) + count
                tally.setdefault(category, {}).setdefault(function, 0)
                tally[category][function] += count
        room.function_votes = votes
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        room.function = ranked[0][0] if ranked and ranked[0][1] >= 2 else _room_type_function(room.type)
    scene.room_functions = {room.instanceid: room.function for room in scene.rooms}
    scene.category_functions = {
        category: sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for category, votes in tally.items()
    }
    scene.categories = {}
    for instance in scene.instances:
        if not instance.valid:
            continue
        name = instance.furniture.category_key or "other"
        scene.categories[name] = scene.categories.get(name, 0) + 1
    return scene


# --------------------------------------------------------------------------- #
# scene -> scene_spec bridge
# --------------------------------------------------------------------------- #
def shell_wall_height(scene, default=DEFAULT_WALL_HEIGHT) -> float:
    """Ceiling height in metres: the tallest ceiling (or wall) mesh vertex."""
    candidates = []
    for mesh in scene.meshes.values():
        if mesh.kind in ("ceiling", "wall"):
            candidates.extend(point[1] for point in mesh.points())
    if not candidates:
        candidates = [point[1] for mesh in scene.meshes.values() for point in mesh.points()]
    return max(candidates) if candidates else default


def camera_specs(scene, source="room", eye_height=DEFAULT_EYE_HEIGHT, max_cameras=0):
    """Synthesize the viewpoint list for the replay.

    ``room`` (default): one camera per room, at the room's representative floor
    point at ``eye_height``.  ``snapshots``: use the positions of the dataset's
    own ``extension.snapshots`` that fall inside a room (falling back to room
    cameras for rooms without one).

    Yaw is deliberately **0** for every camera: the PanoWorld contract fixes the
    camera frame to ``spec.R0`` (yaw 0 looks along world +X) and the validator
    checks the exported extrinsics against exactly that, so a snapshot's looking
    direction is reported in ``source_snapshot`` but never rotated in.
    """
    cameras = []
    used_rooms = set()
    if source == "snapshots":
        for index, snapshot in enumerate(scene.snapshots):
            front = snapshot.get("pos")
            if not front or len(front) < 3:
                continue
            point = front_to_blender_point(front)
            room = scene.in_room(point[:2])
            if room is None or not room.polygon:
                continue
            if room.instanceid in used_rooms:
                continue
            used_rooms.add(room.instanceid)
            cameras.append({
                "name": f"{len(cameras):04d}",
                "room": room.instanceid,
                "pos": [point[0], point[1], point[2]],
                "source_snapshot": index,
            })
    for room in scene.rooms:
        if not room.polygon or room.instanceid in used_rooms or room.catch_all:
            continue
        point = room.camera_point or room.centroid
        if point is None:
            continue
        used_rooms.add(room.instanceid)
        cameras.append({
            "name": f"{len(cameras):04d}",
            "room": room.instanceid,
            "pos": [float(point[0]), float(point[1]), float(eye_height)],
            "source_snapshot": None,
        })
    if max_cameras:
        cameras = cameras[:max_cameras]
    return cameras


def to_scene_spec(scene, model_root, texture_root=None, camera_source="room",
                  eye_height=DEFAULT_EYE_HEIGHT, wall_thickness=DEFAULT_WALL_THICKNESS,
                  max_cameras=0, include_invalid=False):
    """Bridge a parsed 3D-FRONT scene into a ``scene_spec`` dict.

    The result is a normal v2 spec (``spec.parse_spec`` validates it) plus a
    ``front3d`` block that tells the Blender side to replay the dataset shell
    instead of extruding polygons, and ``front3d:<jid>`` furniture references
    that place the dataset's own models verbatim.  Room polygons are recovered
    from the shell's floor meshes, so the top-down drawing and the geometric
    validation keep working on a real dataset house.
    """
    rooms = []
    for room in scene.rooms:
        if not room.polygon:
            continue
        rooms.append({
            "id": room.instanceid,
            "type": room.function,
            "polygon": [[round(float(p[0]), 6), round(float(p[1]), 6)] for p in room.polygon],
            "floor_color": _floor_color(scene, room),
            "source_type": room.type,
            "source_size_m2": room.size,
            "polygon_source": room.polygon_source,
            "catch_all": room.catch_all,
            "categories": room.categories(scene),
        })

    furniture = []
    moved = []
    dropped = []
    for instance in scene.instance_records(include_invalid=include_invalid):
        item = instance.furniture
        pos, height = instance.spec_pos
        room = scene.room_by_id(instance.room_id)
        # spec.validate_spec refuses furniture outside its room polygon, so an
        # out-of-outline item is re-homed to the smallest room that does contain
        # it, or dropped with a note (both happen on real data: doors and
        # windows are dataset furniture items that sit on a wall plane)
        if room is None or not room.polygon or not point_in_polygon(pos, room.polygon):
            other = scene.in_room(pos)
            if other is not None and other.polygon and point_in_polygon(pos, other.polygon):
                moved.append({
                    "uid": item.uid, "reason": "outside its own room outline",
                    "room": instance.room_id, "reassigned_to": other.instanceid,
                    "pos": [round(pos[0], 6), round(pos[1], 6)],
                })
                room = other
            else:
                dropped.append({
                    "uid": item.uid, "reason": "outside every room outline",
                    "room": instance.room_id, "pos": [round(pos[0], 6), round(pos[1], 6)],
                })
                continue
        furniture.append({
            "asset": f"{FURNITURE_SCHEME}:{item.jid}",
            "room": room.instanceid,
            "pos": [round(pos[0], 6), round(pos[1], 6)],
            "rot_z_deg": round(instance.yaw_deg, 9),
            "z": round(height, 6),
            "scale": 1.0,
            "scale_xyz": [round(float(v), 9) for v in instance.scale],
            "mount": "floor",
            "name": item.uid,
            "title": item.title,
            "category": item.category,
            "rot_quat": [float(v) for v in instance.rot],
            "source_pos": [float(v) for v in instance.pos],
            "source_room": instance.room_id,
        })

    cameras = camera_specs(scene, source=camera_source, eye_height=eye_height,
                          max_cameras=max_cameras)
    spec = {
        "_comment": [
            "Generated by blender_agent/front3d.py from a 3D-FRONT scene JSON.",
            "Frames: the dataset is Y-up metres; this spec is the repo/Blender Z-up frame",
            "(x, -z, y).  Furniture keeps its dataset pose exactly: pos is the model's",
            "base-centre, rot_z_deg the dataset yaw about the up axis, scale_xyz the",
            "dataset's per-axis scale.  Shell geometry is replayed from the inline mesh",
            "triangles (front3d.shell = shell_mesh), never extruded from polygons.",
        ],
        "unit": "meter",
        "wall_height": round(shell_wall_height(scene), 6),
        "wall_thickness": float(wall_thickness),
        "rooms": rooms,
        "openings": [],
        "windows": [],
        "cameras": cameras,
        "style": {
            "sun_elevation_deg": 35,
            "sun_azimuth_deg": 210,
            "sky_strength": 1.0,
            "sun_energy": 3.0,
            "view_transform": "AgX",
            "exposure": 1.4,
        },
        "materials": {},
        "details": {"skirting": False, "casings": False, "bevels": False, "glass": False},
        # portals are derived from the dataset's own Window meshes (front3d.py
        # window_lights); the P0/P1 spec would instead list `windows` entries
        "lighting": {"window_portal_w": 42.0, "lamp_w": 0.0, "portal_margin": 0.02},
        "furniture": furniture,
        FURNITURE_SCHEME: {
            "scene_json": scene.path,
            "model_root": os.path.abspath(model_root) if model_root else None,
            "texture_root": os.path.abspath(texture_root) if texture_root else None,
            "scene_uid": scene.uid,
            "scene_version": scene.version,
            "shell": "shell_mesh",
            "up_axis": "Y",
            "front_to_blender": [list(row) for row in FRONT_TO_BLENDER],
            "camera_source": camera_source,
            "source": {
                "rooms": len(scene.rooms),
                "meshes": len(scene.meshes),
                "materials": len(scene.materials),
                "furniture": len(scene.furniture),
                "instances": len(scene.instances),
                "instances_invalid": sum(1 for i in scene.instances if not i.valid),
                "instances_placed": len(furniture),
                "instances_dropped": len(dropped),
                "instances_reassigned": len(moved),
                "lights": len(scene.lights),
                "wall_height_m": round(shell_wall_height(scene), 4),
            },
            "category_functions": scene.category_functions,
            "skipped_furniture": dropped,
            "reassigned_furniture": moved,
            "parse_warnings": scene.warnings,
        },
    }
    return spec


def _floor_color(scene, room):
    """Average floor-mesh material colour, else the repo's warm default."""
    colors = []
    for uid in room.mesh_uids:
        mesh = scene.meshes[uid]
        if mesh.kind != "floor" or mesh.material_uid is None:
            continue
        material = scene.materials.get(mesh.material_uid)
        if material is not None and not material.is_white:
            colors.append(material.rgb)
    if not colors:
        return [0.55, 0.45, 0.34]
    return [round(sum(c[i] for c in colors) / len(colors), 4) for i in range(3)]


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def summary(scene, model_root=None, texture_root=None) -> str:
    lines = []
    lines.append(f"scene {scene.uid}  (version {scene.version})  {scene.path}")
    lines.append(
        f"  rooms {len(scene.rooms)}  meshes {len(scene.meshes)}  materials {len(scene.materials)}  "
        f"furniture models {len(scene.furniture)}  instances {len(scene.instances)}  lights {len(scene.lights)}"
    )
    kinds = {}
    types = {}
    triangles = 0
    vertices = 0
    for mesh in scene.meshes.values():
        kinds[mesh.kind] = kinds.get(mesh.kind, 0) + 1
        label = mesh.type.strip() or "(empty)"
        types[label] = types.get(label, 0) + 1
        triangles += mesh.triangle_count
        vertices += mesh.vertex_count
    lines.append(
        f"  shell: {vertices} vertices, {triangles} triangles, kinds "
        + " ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
    )
    top_types = sorted(types.items(), key=lambda item: -item[1])[:8]
    lines.append("  mesh types: " + " ".join(f"{name}={count}" for name, count in top_types))
    lines.append(f"  wall height (max ceiling y): {shell_wall_height(scene):.3f} m")

    with_texture = sum(1 for m in scene.materials.values() if m.texture_ids)
    colored = sum(1 for m in scene.materials.values() if not m.is_white and not m.texture_ids)
    lines.append(f"  materials: {with_texture} with a texture reference, {colored} with a non-white colour")
    if texture_root:
        found = missing = 0
        for material in scene.materials.values():
            if not material.texture_ids:
                continue
            if resolve_texture(texture_root, material):
                found += 1
            else:
                missing += 1
        lines.append(f"  textures under {texture_root}: {found} found, {missing} missing")

    lines.append("  rooms:")
    for room in scene.rooms:
        categories = room.categories(scene)
        top = ", ".join(f"{name}x{count}" for name, count in sorted(categories.items(), key=lambda kv: -kv[1])[:4])
        lines.append(
            f"    {room.instanceid:<28s} {room.type:<18s} -> {room.function:<14s} "
            f"area {room.area:7.2f} m2  polygon {room.polygon_source:<14s} "
            f"({len(room.polygon)} verts)  cameras {1 if room.camera_point else 0}  {top}"
        )

    lines.append("  furniture categories: " + ", ".join(
        f"{name}x{count}" for name, count in sorted(scene.categories.items(), key=lambda kv: -kv[1])
    ))
    if scene.category_functions:
        lines.append("  category -> room function: " + ", ".join(
            f"{name}->{function}" for name, function in sorted(scene.category_functions.items())
        ))

    missing_models = []
    if model_root:
        seen = set()
        for instance in scene.instances:
            if not instance.valid or instance.jid in seen:
                continue
            seen.add(instance.jid)
            folder = os.path.join(model_root, instance.jid)
            if not os.path.isfile(os.path.join(folder, "raw_model.obj")) and \
               not os.path.isfile(os.path.join(folder, "raw_model_tri.obj")):
                missing_models.append((instance.jid, instance.furniture.title))
        lines.append(
            f"  models under {model_root}: {len(seen) - len(missing_models)}/{len(seen)} present"
        )
        for jid, title in missing_models:
            lines.append(f"    MISSING {jid}  {title}")
    if scene.warnings:
        lines.append(f"  warnings ({len(scene.warnings)}):")
        for warning in scene.warnings[:20]:
            lines.append(f"    - {warning}")
        if len(scene.warnings) > 20:
            lines.append(f"    ... {len(scene.warnings) - 20} more")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Blender side: material + shell + furniture import
# --------------------------------------------------------------------------- #
def _bpy():
    import bpy  # noqa: F401  (lazy: the host CLI must work without Blender)

    return bpy


SHELL_COLLECTION = "front3d_shell"
FURNITURE_COLLECTION = "front3d_furniture"
LIBRARY_PREFIX = "front3d_model_"

# fallbacks for materials that ship a white colour and no texture (very common
# in 3D-FRONT scenes): the flat P0 look, per shell kind
FALLBACK_COLORS = {
    "floor": (0.55, 0.45, 0.34),
    "ceiling": (0.95, 0.95, 0.95),
    "wall": (0.90, 0.87, 0.82),
    "window": (0.75, 0.85, 0.92),
    "other": (0.82, 0.81, 0.78),
}
FALLBACK_ROUGHNESS = {"floor": 0.6, "ceiling": 0.9, "wall": 0.9, "window": 0.12, "other": 0.8}


def _fallback_key(mesh_type: str) -> str:
    lowered = (mesh_type or "").lower()
    if "window" in lowered or "glass" in lowered:
        return "window"
    if "floor" in lowered:
        return "floor"
    if "ceil" in lowered:
        return "ceiling"
    if "wall" in lowered:
        return "wall"
    return "other"


def _flat_material(name, color, roughness):
    bpy = _bpy()
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    bsdf.inputs["Roughness"].default_value = float(roughness)
    bsdf.inputs["Metallic"].default_value = 0.0
    return material


def glass_material(name="front3d_glass", tint=(0.92, 0.96, 0.98)):
    """Transmissive glass for the dataset's ``Window`` meshes.

    A 3D-FRONT shell's window pane is a plain quad; with an opaque material the
    room would be lit by the world sky only through whatever gap is left, which
    reads as a black room.  Transmission is what makes the pane behave like
    glass (the same as ``details.make_glass_material`` for the P0/P1 shell).
    """
    bpy = _bpy()
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (float(tint[0]), float(tint[1]), float(tint[2]), 1.0)
    bsdf.inputs["Roughness"].default_value = 0.02
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["IOR"].default_value = 1.45
    bsdf.inputs["Metallic"].default_value = 0.0
    return material


def _image_material(name, image_path):
    """Principled material driven by one image texture (linear colour space)."""
    bpy = _bpy()
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.6
    image = bpy.data.images.get(os.path.basename(image_path))
    if image is None:
        image = bpy.data.images.load(image_path, check_existing=True)
    image.colorspace_settings.name = "sRGB"
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def resolve_texture(texture_root, material):
    """Path of the material's texture in ``3D-FRONT-textures``, or ``None``.

    Both folder-name rules are tried (see ``Front3DMaterial.texture_ids``).
    BlenderProc downloads a missing texture from the material's URL; this module
    never goes to the network -- a missing file falls back to the flat colour.
    """
    if not texture_root or material is None:
        return None
    for texture_id in material.texture_ids:
        path = os.path.join(texture_root, texture_id, "texture.png")
        if os.path.isfile(path):
            return path
    return None


def shell_material_cache(scene, texture_root, cache=None, report=None):
    """One Blender material per dataset material uid (deduplicated).

    Order of preference: the material's own 3D-FRONT texture (if the folder was
    handed in), then its flat colour, then a per-kind flat fallback for the very
    common "white colour, no texture" case.  Ceilings are *not* made emissive
    (BlenderProc does that to light its scenes); this repo lights the room
    physically through the stage, and a self-lit ceiling would double the sky.
    """
    cache = {} if cache is None else cache
    report = {} if report is None else report
    for key, _ in (("textures_used", 0), ("colors_used", 0), ("fallbacks_used", 0)):
        report.setdefault(key, 0)
    for uid, material in scene.materials.items():
        key = ("dataset", uid)
        if key in cache:
            continue
        name = "front3d_" + "".join(ch if ch.isalnum() else "_" for ch in uid)
        image_path = resolve_texture(texture_root, material)
        if image_path:
            cache[key] = _image_material(name, image_path)
            report["textures_used"] += 1
        elif not material.is_white:
            cache[key] = _flat_material(name, material.rgb, 0.6)
            report["colors_used"] += 1
        else:
            cache[key] = None      # decide per mesh kind below
            report["fallbacks_used"] += 1
    for kind, color in FALLBACK_COLORS.items():
        key = ("fallback", kind)
        if key not in cache:
            cache[key] = _flat_material(f"front3d_fallback_{kind}", color, FALLBACK_ROUGHNESS[kind])
    report["materials"] = len(cache)
    return cache


def shell_material_for(cache, mesh, dataset_materials=None):
    """Blender material for one dataset mesh (windows get real glass)."""
    if _fallback_key(mesh.type) == "window":
        key = ("glass", mesh.material_uid or "default")
        if key not in cache:
            tint = (0.92, 0.96, 0.98)
            entry = (dataset_materials or {}).get(mesh.material_uid)
            if entry is not None and not entry.is_white:
                tint = tuple(entry.rgb)
            name = "front3d_glass_" + "".join(
                ch if ch.isalnum() else "_" for ch in (mesh.material_uid or "default")
            )
            cache[key] = glass_material(name, tint=tint)
        return cache[key]
    key = ("dataset", mesh.material_uid) if mesh.material_uid else None
    material = cache.get(key) if key else None
    if material is None:
        material = cache[("fallback", _fallback_key(mesh.type))]
    return material


def _mesh_data_from_front3d(mesh, name):
    """Build a Blender mesh from one dataset ``mesh`` entry (front frame).

    Vertices are converted with ``M = FRONT_TO_BLENDER`` here, so the object
    itself stays at the identity transform.  Dataset normals are not imported
    (they are unreliable on shell meshes -- a wall quad in the real data claims
    ``[0, 0, 1]``); Blender recomputes them from the winding.
    """
    bpy = _bpy()
    transform = getattr(mesh, "child_transform", None)
    points = mesh.points()
    if transform is not None:
        matrix = scene_instance_matrix(*transform)
        points = [tuple(mat4_apply(matrix, point)) for point in points]
    vertices = [tuple(front_to_blender_point(point)) for point in points]
    faces = [
        (mesh.faces[i], mesh.faces[i + 1], mesh.faces[i + 2])
        for i in range(0, len(mesh.faces), 3)
    ]
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    if mesh.uvs is not None and len(mesh.uvs) >= 2 * len(vertices):
        data.uv_layers.new(name="UVMap")
        values = mesh.uvs
        # dataset UVs are stored per vertex and shared between the faces using it
        uv_data = data.uv_layers.active.data
        for polygon in data.polygons:
            for loop_index in polygon.loop_indices:
                vertex = data.loops[loop_index].vertex_index
                uv_data[loop_index].uv = (values[2 * vertex], values[2 * vertex + 1])
    data.validate()
    data.update()
    return data


def build_shell(block, report=None):
    """Replay the dataset shell: every ``mesh[]`` entry becomes one object.

    Mirrors ``shell.build_shell``'s return shape (``floors`` / ``ceilings`` /
    ``walls`` / ``shell_objects``) so the manifest, the sky dome and the PLY
    export keep working.  Returns ``(shell_info, report)``.
    """
    bpy = _bpy()
    from blender_agent import shell as shell_mod

    scene = parse_scene(block["scene_json"], model_root=block.get("model_root"))
    collection = shell_mod.ensure_collection(SHELL_COLLECTION)
    report = {} if report is None else report
    cache = shell_material_cache(scene, block.get("texture_root"), report=report)

    objects = []
    by_kind = {"floor": [], "ceiling": [], "wall": [], "other": []}
    triangles = 0
    for mesh in scene.meshes.values():
        name = "front3d_" + "".join(ch if ch.isalnum() else "_" for ch in mesh.uid)
        if mesh.triangle_count == 0 or mesh.vertex_count == 0:
            report.setdefault("skipped_meshes", []).append(mesh.uid)
            continue
        data = _mesh_data_from_front3d(mesh, name)
        obj = bpy.data.objects.new(name, data)
        obj["front3d_uid"] = mesh.uid
        obj["front3d_type"] = mesh.type
        obj.data.materials.append(shell_material_for(cache, mesh, scene.materials))
        collection.objects.link(obj)
        objects.append(obj)
        by_kind[mesh.kind].append(obj)
        triangles += mesh.triangle_count

    report.update({
        "scene_uid": scene.uid,
        "scene_version": scene.version,
        "meshes": len(objects),
        "triangles": triangles,
        "by_kind": {kind: len(items) for kind, items in by_kind.items()},
        "wall_height_m": round(shell_wall_height(scene), 4),
        "parse_warnings": len(scene.warnings),
        "room_functions": scene.room_functions,
        "category_functions": scene.category_functions,
        "window_meshes": sum(1 for mesh in scene.meshes.values()
                             if _fallback_key(mesh.type) == "window"),
    })
    shell_info = {
        "collection": collection,
        "floors": by_kind["floor"],
        "ceilings": by_kind["ceiling"],
        "walls": by_kind["wall"],
        "others": by_kind["other"],
        "shell_objects": objects,
        "cutters_applied": {},
        "front3d": report,
        "front3d_scene": scene,     # in-memory only: window lights need it
    }
    return shell_info, report


def clear_parent_keep_transform(obj):
    if obj.parent is None:
        return
    matrix = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = matrix


def window_lights(scene, energy, collection=None, margin: float = 0.02, report=None):
    """Soft area lights just inside every dataset ``Window`` pane, aimed inward.

    P0/P1 derive their portals from the spec's ``windows`` entries, which a
    dataset shell does not have; a 3D-FRONT scene instead carries window *meshes*,
    and those give both the pane rectangle and -- via the room that owns the pane
    -- the direction into the room.  Same purpose as
    ``stage.build_window_portals``: the glass already lets the world sky in, but a
    large soft source inside the reveal is what makes a 128-sample render clean.
    """
    bpy = _bpy()
    import mathutils

    lights = []
    details = []
    report = {} if report is None else report
    for index, mesh in enumerate(scene.meshes.values()):
        if _fallback_key(mesh.type) != "window":
            continue
        points = [front_to_blender_point(point) for point in mesh.points()]
        if len(points) < 3:
            continue
        count = len(points)
        centroid = [sum(point[i] for point in points) / count for i in range(3)]
        # pane frame: the longest edge is u, the polygon normal gives v and n
        best = None
        for i in range(count):
            a, b = points[i], points[(i + 1) % count]
            edge = [b[k] - a[k] for k in range(3)]
            length = math.sqrt(sum(value * value for value in edge))
            if best is None or length > best[0]:
                best = (length, edge)
        if best is None or best[0] < 1e-6:
            continue
        u = [value / best[0] for value in best[1]]
        normal = [0.0, 0.0, 0.0]
        for i in range(count):
            a, b, c = points[i], points[(i + 1) % count], points[(i + 2) % count]
            first = [b[k] - a[k] for k in range(3)]
            second = [c[k] - b[k] for k in range(3)]
            cross = [first[1] * second[2] - first[2] * second[1],
                     first[2] * second[0] - first[0] * second[2],
                     first[0] * second[1] - first[1] * second[0]]
            normal = [normal[k] + cross[k] for k in range(3)]
        norm = math.sqrt(sum(value * value for value in normal)) or 1.0
        normal = [value / norm for value in normal]
        v = [normal[1] * u[2] - normal[2] * u[1],
             normal[2] * u[0] - normal[0] * u[2],
             normal[0] * u[1] - normal[1] * u[0]]
        # an area light's ``size`` is its width and ``size_y`` its height, so the
        # pane's more horizontal in-plane axis has to be ``u`` (a tall pane's
        # longest edge is vertical)
        if math.hypot(u[0], u[1]) < math.hypot(v[0], v[1]):
            u, v = v, u
        spans_u = [sum((point[k] - centroid[k]) * u[k] for k in range(3)) for point in points]
        spans_v = [sum((point[k] - centroid[k]) * v[k] for k in range(3)) for point in points]
        size_u = max(spans_u) - min(spans_u)
        size_v = max(spans_v) - min(spans_v)

        room = scene.in_room((centroid[0], centroid[1]))
        target = room.camera_point if room is not None else None
        if target is None:
            continue
        inward = [float(target[0]) - centroid[0], float(target[1]) - centroid[1], 0.0]
        length = math.hypot(inward[0], inward[1])
        if length < 1e-6:
            continue
        inward = [inward[0] / length, inward[1] / length, 0.0]

        name = f"front3d_window_portal_{index}"
        width = max(0.2, size_u - margin)
        height = max(0.2, size_v - margin)
        light_data = bpy.data.lights.new(name, type="AREA")
        light_data.shape = "RECTANGLE"
        light_data.size = width
        light_data.size_y = height
        light_data.energy = float(energy)
        light_data.color = (0.96, 0.98, 1.0)
        light = bpy.data.objects.new(name, light_data)
        light.location = [centroid[k] + inward[k] * margin for k in range(3)]
        light.rotation_euler = mathutils.Vector(inward).to_track_quat("-Z", "Y").to_euler()
        light.visible_camera = False          # lighting gear, not geometry
        light["front3d_window_mesh"] = mesh.uid
        light["front3d_room"] = room.instanceid
        (collection or bpy.context.scene.collection).objects.link(light)
        lights.append(light)
        details.append({
            "mesh": mesh.uid,
            "room": room.instanceid,
            "position": [round(float(v), 6) for v in light.location],
            "inward": [round(float(v), 6) for v in inward],
            "size_m": [round(float(width), 6), round(float(height), 6)],
            "pane_m": [round(float(size_u), 6), round(float(size_v), 6)],
        })
    report["window_lights"] = len(lights)
    report["window_light_details"] = details
    return lights


def import_model(model_root, jid, cache=None, report=None):
    """Import ``<model_root>/<jid>/raw_model.obj`` once; returns its library dict.

    The OBJ is imported with ``forward_axis="Y", up_axis="Z"`` so Blender adds
    **no** axis conversion: the vertices stay in the dataset's Y-up frame and the
    instance matrix (``M @ T @ R @ S``) is the only transform applied, exactly
    like ``transform_v`` in the official toolbox.
    """
    bpy = _bpy()
    cache = {} if cache is None else cache
    report = {} if report is None else report
    if jid in cache:
        return cache[jid]

    folder = os.path.join(model_root, jid)
    source = None
    for candidate in ("raw_model.obj", "raw_model_tri.obj"):
        path = os.path.join(folder, candidate)
        if os.path.isfile(path):
            source = path
            break
    if source is None:
        cache[jid] = None
        report.setdefault("missing_models", []).append(jid)
        return None

    before = set(bpy.data.objects)
    # no axis conversion on purpose (see the docstring): the dataset pose is
    # applied afterwards, to raw Y-up model vertices
    bpy.ops.wm.obj_import(filepath=source, forward_axis="Y", up_axis="Z")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    library = bpy.data.collections.new(LIBRARY_PREFIX + jid)
    vertices = polygons = 0
    for obj in meshes:
        clear_parent_keep_transform(obj)
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)
        library.objects.link(obj)
        vertices += len(obj.data.vertices)
        polygons += len(obj.data.polygons)
    for obj in imported:
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)

    material = model_material(jid, folder, report)
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(material)

    points = [tuple(v.co) for obj in meshes for v in obj.data.vertices]
    bbox = bbox_of_points(points)
    entry = {
        "jid": jid,
        "source": source,
        "collection": library,
        "objects": meshes,
        "bbox": bbox,
        "local_bbox": [[round(v, 6) for v in bbox[0]], [round(v, 6) for v in bbox[1]]] if bbox else None,
        "vertices": vertices,
        "polygons": polygons,
        "material": material.name,
    }
    report.setdefault("models_imported", []).append({"jid": jid, "vertices": vertices,
                                                     "polygons": polygons, "source": os.path.basename(source)})
    cache[jid] = entry
    return entry


def model_material(jid, folder, report=None):
    """Material for one 3D-FUTURE model: ``texture.png`` if present, else mtl/flat.

    BlenderProc zeroes the emission and transmission the dataset's ``model.mtl``
    files carry (they make everything glow / turn glassy), then puts
    ``texture.png`` on the base colour; the same is done here.
    """
    bpy = _bpy()
    report = {} if report is None else report
    texture_path = os.path.join(folder, "texture.png")
    name = "front3d_model_" + "".join(ch if ch.isalnum() else "_" for ch in jid)
    if os.path.isfile(texture_path):
        report["model_textures"] = report.get("model_textures", 0) + 1
        return _image_material(name, texture_path)
    report["model_texture_missing"] = report.get("model_texture_missing", 0) + 1
    return _flat_material(name + "_fallback", (0.7, 0.68, 0.65), 0.7)


def item_matrix(pos_xy, height, rot_z_deg, scale_xyz):
    """World matrix for one ``front3d:<jid>`` spec item (repo-frame pose).

    The bridge stores the dataset transform in this repo's Z-up frame, so the
    front-frame position is recovered first and the dataset composition is
    rebuilt exactly: ``M @ T(pos_front) @ R_y(yaw) @ S(scale_xyz)``.  The yaw
    about the dataset's up axis becomes a quaternion about +Y, which is what the
    dataset itself stores (``(0, sin(yaw/2), 0, cos(yaw/2))``).
    """
    front_pos = blender_to_front_point([float(pos_xy[0]), float(pos_xy[1]), float(height)])
    half = math.radians(float(rot_z_deg)) / 2.0
    quat = (0.0, math.sin(half), 0.0, math.cos(half))
    return scene_instance_matrix(front_pos, quat, scale_xyz, frame=True)


def place_model(model, name, matrix, collection=None, properties=None):
    """Drop a collection instance of an imported model at ``matrix``.

    The dataset pose is used verbatim -- no floor re-anchoring: the instance is
    placed exactly where ``pos``/``rot``/``scale`` put it, and the model's own
    base-centre origin (verified on real data) is what makes it stand on the
    floor.  Instancing shares mesh and texture data between repeats of a model.
    """
    bpy = _bpy()
    empty = bpy.data.objects.new(name, None)
    empty.instance_type = "COLLECTION"
    empty.instance_collection = model["collection"]
    empty.empty_display_size = 0.15
    for i in range(4):
        for j in range(4):
            empty.matrix_world[i][j] = float(matrix[i][j])
    for key, value in (properties or {}).items():
        empty[key] = value
    (collection or bpy.context.scene.collection).objects.link(empty)
    return empty


def model_world_bbox(model, matrix):
    """Exact world AABB of a placed model (its 8 transformed local corners)."""
    if not model or not model.get("bbox"):
        return None
    bbox = transform_bbox(matrix, model["bbox"])
    return [[round(v, 6) for v in bbox[0]], [round(v, 6) for v in bbox[1]]]


# --------------------------------------------------------------------------- #
# command line (host side)
# --------------------------------------------------------------------------- #
def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="front3d.py",
        description="Parse a 3D-FRONT scene JSON (host side) and summarize / bridge it to a scene spec",
    )
    parser.add_argument("--scene", required=True, help="path to the 3D-FRONT scene JSON")
    parser.add_argument("--model-root", default=None, help="path to 3D-FUTURE-model")
    parser.add_argument("--texture-root", default=None, help="path to 3D-FRONT-textures")
    parser.add_argument("--dry-run", action="store_true", help="print the parse summary and exit")
    parser.add_argument("--spec-out", default=None, help="write the bridged scene_spec JSON here")
    parser.add_argument("--camera-source", default="room", choices=["room", "snapshots"],
                        help="where the replay gets its viewpoints from")
    parser.add_argument("--eye-height", type=float, default=DEFAULT_EYE_HEIGHT)
    parser.add_argument("--wall-thickness", type=float, default=DEFAULT_WALL_THICKNESS)
    parser.add_argument("--max-cameras", type=int, default=0, help="0 = one per room")
    parser.add_argument("--include-invalid", action="store_true",
                        help="also bridge furniture flagged 'valid: false'")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not os.path.isfile(args.scene):
        print(f"front3d: no such scene JSON: {args.scene}", file=sys.stderr)
        return 2
    scene = parse_scene(args.scene, model_root=args.model_root)

    if args.json:
        payload = {
            "uid": scene.uid,
            "version": scene.version,
            "rooms": [
                {"instanceid": room.instanceid, "type": room.type, "function": room.function,
                 "area_m2": round(room.area, 4), "polygon_source": room.polygon_source,
                 "categories": room.categories(scene)}
                for room in scene.rooms
            ],
            "meshes": len(scene.meshes),
            "materials": len(scene.materials),
            "furniture_categories": scene.categories,
            "category_functions": scene.category_functions,
            "instances": len(scene.instances),
            "wall_height_m": round(shell_wall_height(scene), 4),
            "warnings": scene.warnings,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(summary(scene, model_root=args.model_root, texture_root=args.texture_root))

    if args.spec_out:
        spec = to_scene_spec(
            scene,
            model_root=args.model_root,
            texture_root=args.texture_root,
            camera_source=args.camera_source,
            eye_height=args.eye_height,
            wall_thickness=args.wall_thickness,
            max_cameras=args.max_cameras,
            include_invalid=args.include_invalid,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.spec_out)), exist_ok=True)
        with open(args.spec_out, "w", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=2)
            handle.write("\n")
        print(
            f"front3d: wrote {args.spec_out} "
            f"({len(spec['rooms'])} rooms, {len(spec['furniture'])} furniture, {len(spec['cameras'])} cameras)"
        )
    if args.dry_run or args.spec_out:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
