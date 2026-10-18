"""Scene inventory and per-floor collision meshes for navigation.

The navigation stack consumes the same delivered build as the renderer: the
furnished PLY when the build has one (shell + details + furniture + sky dome),
otherwise the shell PLY, otherwise ``scene.glb``.  The merged mesh is filtered
to one storey's vertical band and written as a small Y-up OBJ that Recast can
bake without seeing the storey above.  Filtering before the bake is what makes
the upper floor reachable: Recast drops one of two overlapping floor slabs when
both are in the same mesh, while a per-storey OBJ keeps each slab alone.

Coordinate frames
-----------------
Build PLYs are Blender Z-up metres; GLBs are glTF Y-up metres.  Internally
every mesh is normalised to **Blender Z-up** (that is what the spec elevations
use).  :func:`write_floor_obj` converts to the **glTF / Habitat Y-up** frame
``(x, z, -y)``, exactly the conversion ``blender_agent/export_glb.py`` applies
on export.  Habitat-Sim imports the OBJ as Y-up, so a storey's walking surface
lands at ``y = elevation`` (Recast's cell height adds a few millimetres to the
baked surface, handled in ``navmesh.py``).

The inventory is explicit, not a glob: a re-run must use the same 20 scenes and
the same built artifacts even when newer outputs appear in ``outputs/``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import spec as spec_mod  # noqa: E402

#: never treat this many metres outside the room outline as scene content
BOUNDS_MARGIN_M = 1.0
#: slab top/bottom triangles are normal scene content; this guards float noise
BAND_EPS_M = 0.01


@dataclass(frozen=True)
class SceneEntry:
    """One built scene used by the navigation benchmark."""

    scene_id: str
    spec_path: str
    build_dir: str
    #: optional explicit mesh (``.ply`` / ``.glb``); default searches the build
    mesh_path: Optional[str] = None

    def resolved(self, root: str = REPO_ROOT) -> "SceneEntry":
        def path(value: Optional[str]) -> Optional[str]:
            if value is None or os.path.isabs(value):
                return value
            return os.path.join(root, value)

        return SceneEntry(
            scene_id=self.scene_id,
            spec_path=path(self.spec_path) or self.spec_path,
            build_dir=path(self.build_dir) or self.build_dir,
            mesh_path=path(self.mesh_path),
        )


def _entry(scene_id: str, base: str, spec: str = "spec.json",
           build: str = "build", mesh: Optional[str] = None) -> SceneEntry:
    return SceneEntry(
        scene_id=scene_id,
        spec_path=os.path.join(base, spec),
        build_dir=os.path.join(base, build),
        mesh_path=mesh,
    )


#: 20 delivered scenes: 10 benchmark builds, 6 acceptance-family builds,
#: 3 cross-level acceptance scenes (stairs, ramp / split level, void) and one
#: open-plan build.  All have ``build/build_manifest.json`` and a PLY.
DEFAULT_SCENES: Tuple[SceneEntry, ...] = (
    _entry("bench_scene_0000", "outputs/benchmark100/sample/scene_0000"),
    _entry("bench_scene_0001", "outputs/benchmark100/sample/scene_0001"),
    _entry("bench_scene_0002", "outputs/benchmark100/sample/scene_0002"),
    _entry("bench_scene_0003", "outputs/benchmark100/sample/scene_0003"),
    _entry("bench_scene_0005", "outputs/benchmark100/sample/scene_0005"),
    _entry("bench_scene_0008", "outputs/benchmark100/sample/scene_0008"),
    _entry("bench_scene_0048", "outputs/benchmark100/sample/scene_0048"),
    _entry("bench_scene_0049", "outputs/benchmark100/sample/scene_0049"),
    _entry("bench_scene_0092", "outputs/benchmark100/sample/scene_0092"),
    _entry("bench_scene_0095", "outputs/benchmark100/sample/scene_0095"),
    _entry("family_auto_seed11", "outputs/acceptance_families/auto_seed11"),
    _entry("family_central_living_seed0",
           "outputs/acceptance_families/central_living_seed0"),
    _entry("family_compact_2x2_seed0",
           "outputs/acceptance_families/compact_2x2_seed0"),
    _entry("family_corridor_double_seed4",
           "outputs/acceptance_families/corridor_double_seed4"),
    _entry("family_l_shape_seed3", "outputs/acceptance_families/l_shape_seed3"),
    _entry("family_loop_seed0", "outputs/acceptance_families/loop_seed0"),
    _entry("cross_mezzanine", "outputs/blender_agent/cross_level/mezzanine",
           spec="scene_spec.json"),
    _entry("cross_split_level", "outputs/blender_agent/cross_level/split_level",
           spec="scene_spec.json"),
    _entry("cross_void", "outputs/blender_agent/cross_level/void",
           spec="scene_spec.json", build="e2e/build"),
    _entry("layout_open_plan_seed0002",
           "outputs/layout_gen/open_plan_seed0002"),
)


def default_scenes(root: str = REPO_ROOT) -> List[SceneEntry]:
    return [entry.resolved(root) for entry in DEFAULT_SCENES]


def find_scene(scene_id: str, root: str = REPO_ROOT) -> SceneEntry:
    for entry in default_scenes(root):
        if entry.scene_id == scene_id:
            return entry
    raise KeyError(f"unknown scene id '{scene_id}'")


# --------------------------------------------------------------------------- #
# mesh containers
# --------------------------------------------------------------------------- #
@dataclass
class Mesh:
    """Triangle mesh in the internal Blender Z-up frame (metres)."""

    vertices: np.ndarray  # (N, 3) float64
    triangles: np.ndarray  # (M, 3) int64
    source: str = ""

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.triangles = np.asarray(self.triangles, dtype=np.int64).reshape(-1, 3)

    def __len__(self) -> int:
        return int(len(self.triangles))

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.vertices) == 0:
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def transformed(self, matrix: np.ndarray) -> "Mesh":
        points = self.vertices @ matrix[:3, :3].T + matrix[:3, 3]
        return Mesh(points, self.triangles.copy(), self.source)

    def subset(self, triangle_indices: np.ndarray) -> "Mesh":
        if len(triangle_indices) == 0:
            return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), self.source)
        triangles = self.triangles[triangle_indices]
        used, inverse = np.unique(triangles.reshape(-1), return_inverse=True)
        return Mesh(self.vertices[used], inverse.reshape(-1, 3), self.source)

    def stats(self) -> dict:
        return {"vertices": int(len(self.vertices)), "triangles": int(len(self.triangles)),
                "source": self.source}


# --------------------------------------------------------------------------- #
# PLY
# --------------------------------------------------------------------------- #
def read_ply(path: str) -> Mesh:
    """Read the binary little-endian triangle PLY written by the builder.

    The builder's ``panorama.write_binary_ply`` emits float32 x/y/z vertices and
    a ``uchar int int int`` face list; that is the whole supported dialect.  A
    clear error is better than guessing at ASCII or arbitrary properties.
    """
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        format_line = handle.readline().strip()
        if format_line != b"format binary_little_endian 1.0":
            raise ValueError(f"{path}: unsupported PLY format '{format_line.decode('ascii', 'replace')}'")
        vertex_count = face_count = None
        properties: List[str] = []
        face_properties: List[str] = []
        in_vertex = False
        in_face = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            text = line.strip()
            if text == b"end_header":
                break
            if text.startswith(b"element vertex "):
                vertex_count = int(text.split()[-1])
                in_vertex, in_face = True, False
            elif text.startswith(b"element face "):
                face_count = int(text.split()[-1])
                in_face, in_vertex = True, False
            elif text.startswith(b"element "):
                in_vertex = in_face = False
            elif text.startswith(b"property ") and in_vertex:
                properties.append(text.split()[-1].decode("ascii"))
            elif text.startswith(b"property list") and in_face:
                face_properties.append(text.decode("ascii"))
        if vertex_count is None or face_count is None:
            raise ValueError(f"{path}: PLY has no vertex/face count")
        if properties[:3] != ["x", "y", "z"]:
            raise ValueError(f"{path}: expected x/y/z vertex properties, got {properties}")
        expected = "property list uchar int vertex_indices"
        if face_properties != [expected]:
            raise ValueError(f"{path}: expected '{expected}', got {face_properties}")

        vertex_bytes = vertex_count * 3 * 4
        raw = handle.read(vertex_bytes)
        if len(raw) != vertex_bytes:
            raise ValueError(f"{path}: truncated vertex block")
        vertices = np.frombuffer(raw, dtype="<f4").reshape(vertex_count, 3).astype(np.float64)

        face_dtype = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
        raw = handle.read(face_count * face_dtype.itemsize)
        if len(raw) != face_count * face_dtype.itemsize:
            raise ValueError(f"{path}: truncated face block")
        faces = np.frombuffer(raw, dtype=face_dtype)
        if len(faces) and not np.all(faces["n"] == 3):
            raise ValueError(f"{path}: only triangle faces are supported")
        triangles = np.stack([faces["a"], faces["b"], faces["c"]], axis=1).astype(np.int64)
    return Mesh(vertices, triangles, source=path)


def write_ply(path: str, mesh: Mesh) -> dict:
    """Write a binary triangle PLY in the internal Blender frame (tests / tools)."""
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(mesh.vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(mesh.triangles)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    face_dtype = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(np.ascontiguousarray(mesh.vertices, dtype="<f4").tobytes())
        block = np.empty(len(mesh.triangles), dtype=face_dtype)
        block["n"] = 3
        block["a"], block["b"], block["c"] = (mesh.triangles[:, 0], mesh.triangles[:, 1],
                                               mesh.triangles[:, 2])
        handle.write(block.tobytes())
    return {"bytes": os.path.getsize(path), **mesh.stats()}


# --------------------------------------------------------------------------- #
# GLB
# --------------------------------------------------------------------------- #
_GLTF_COMPONENTS = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16,
    5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
_GLTF_TYPES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
               "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _glb_chunks(path: str) -> Tuple[dict, bytes]:
    with open(path, "rb") as handle:
        magic, version, length = struct.unpack("<4sII", handle.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError(f"{path}: not a glTF 2.0 binary file")
        document = None
        binary = b""
        while handle.tell() < length:
            chunk_length, chunk_type = struct.unpack("<II", handle.read(8))
            payload = handle.read(chunk_length)
            if chunk_type == 0x4E4F534A:  # JSON
                document = json.loads(payload.decode("utf-8"))
            elif chunk_type == 0x004E4942:  # BIN
                binary = payload
        if document is None:
            raise ValueError(f"{path}: GLB has no JSON chunk")
    return document, binary


def _accessor(document: dict, binary: bytes, index: int):
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(_GLTF_COMPONENTS[accessor["componentType"]]).newbyteorder("<")
    width = _GLTF_TYPES[accessor["type"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", 0)) or dtype.itemsize * width
    count = int(accessor["count"])
    if stride == dtype.itemsize * width:
        array = np.frombuffer(binary, dtype=dtype, count=count * width,
                              offset=offset).reshape(count, width)
    else:
        rows = []
        for i in range(count):
            start = offset + i * stride
            rows.append(np.frombuffer(binary, dtype=dtype, count=width, offset=start))
        array = np.stack(rows) if rows else np.zeros((0, width), dtype=dtype)
    return np.array(array, copy=True)


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        values = node["matrix"]
        return np.array(values, dtype=np.float64).reshape(4, 4).T
    translation = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    rotation = np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    scale = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    x, y, z, w = rotation
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        rotation_matrix = np.eye(3)
    else:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        rotation_matrix = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix * scale[None, :]
    matrix[:3, 3] = translation
    return matrix


def read_glb(path: str) -> Mesh:
    """Read triangle geometry from a GLB, normalised to the Blender Z-up frame."""
    document, binary = _glb_chunks(path)
    accessors = document.get("accessors", [])
    meshes = document.get("meshes", [])
    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    roots = scenes[document.get("scene", 0)].get("nodes", []) if scenes else list(range(len(nodes)))

    chunks: List[Mesh] = []

    def visit(index: int, parent: np.ndarray) -> None:
        node = nodes[index]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            for primitive in meshes[node["mesh"]].get("primitives", []):
                if primitive.get("mode", 4) != 4:
                    continue
                attributes = primitive.get("attributes", {})
                if "POSITION" not in attributes:
                    continue
                positions = _accessor(document, binary, attributes["POSITION"])
                if "indices" in primitive:
                    indices = _accessor(document, binary, primitive["indices"]).reshape(-1)
                else:
                    indices = np.arange(len(positions), dtype=np.int64)
                triangles = indices.reshape(-1, 3)
                # Apply the node hierarchy in its own glTF frame first, then
                # convert Y-up -> Blender Z-up: (x, y, z) -> (x, -z, y).
                gltf_points = positions @ world[:3, :3].T + world[:3, 3]
                points = np.stack([gltf_points[:, 0], -gltf_points[:, 2],
                                   gltf_points[:, 1]], axis=1)
                chunks.append(Mesh(points, triangles))
        for child in node.get("children", []):
            visit(child, world)

    for root in roots:
        visit(root, np.eye(4, dtype=np.float64))
    if not chunks:
        raise ValueError(f"{path}: GLB contains no triangle geometry")
    vertices = np.concatenate([part.vertices for part in chunks], axis=0)
    triangles = np.concatenate(
        [part.triangles + sum(len(prev.vertices) for prev in chunks[:i])
         for i, part in enumerate(chunks)], axis=0)
    return Mesh(vertices, triangles, source=path)


# --------------------------------------------------------------------------- #
# scene / spec access
# --------------------------------------------------------------------------- #
def load_scene(entry: SceneEntry) -> spec_mod.SceneSpec:
    """Load the spec of an inventory entry (embedded in a build manifest if needed)."""
    path = entry.spec_path
    if os.path.basename(path) == "build_manifest.json":
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if "spec" not in manifest:
            raise ValueError(f"{path}: build manifest carries no 'spec' block")
        return spec_mod.parse_spec(manifest["spec"], source_path=path)
    return spec_mod.load_spec(path)


def floors_of(spec: spec_mod.SceneSpec) -> List[spec_mod.FloorSpec]:
    """Storeys of a spec, with a synthetic one for a legacy single-storey scene."""
    if spec.floors:
        return list(spec.floors)
    return [spec_mod.FloorSpec(id="F0", elevation=0.0, height=spec.wall_height,
                               slab_thickness=spec_mod.DEFAULT_SLAB_M)]


def floor_id_of_room(spec: spec_mod.SceneSpec, room_id: str) -> str:
    if spec.floors:
        return spec.room(room_id).floor_id or ""
    return "F0"


def room_floor_id(spec: spec_mod.SceneSpec, room: spec_mod.RoomSpec) -> str:
    return (room.floor_id or "") if spec.floors else "F0"


def scene_bounds(spec: spec_mod.SceneSpec) -> Tuple[np.ndarray, np.ndarray]:
    """XY bounds of the room outlines and the vertical extent of the building."""
    xs: List[float] = []
    ys: List[float] = []
    for room in spec.rooms:
        xs.extend(float(p[0]) for p in room.polygon)
        ys.extend(float(p[1]) for p in room.polygon)
    if not xs or not ys:
        raise ValueError("spec has no room polygons")
    storeys = floors_of(spec)
    z_low = min(floor.slab_bottom for floor in storeys) - 0.2
    z_high = max(spec.room_top(room.id) for room in spec.rooms)
    if spec.floors:
        z_high = max(z_high, max(floor.top for floor in storeys)) + 0.2
    else:
        z_high += spec_mod.DEFAULT_SLAB_M + 0.2
    lo = np.array([min(xs) - BOUNDS_MARGIN_M, min(ys) - BOUNDS_MARGIN_M, z_low])
    hi = np.array([max(xs) + BOUNDS_MARGIN_M, max(ys) + BOUNDS_MARGIN_M, z_high])
    return lo, hi


def floor_band(spec: spec_mod.SceneSpec, floor: spec_mod.FloorSpec) -> Tuple[float, float]:
    """Blender-Z band that holds one storey's shell and furniture.

    The lower bound includes the storey's slab body; the upper bound stops a
    hair below the ceiling plane / upper slab so the slab top (which belongs to
    the storey above) is never baked as a second floor in this storey's mesh.
    Split-level rooms may reach above ``floor.elevation + floor.height``; the
    room tops are included explicitly.
    """
    if spec.floors:
        rooms = spec.floor_rooms(floor.id)
        top = max([floor.elevation + floor.height]
                  + [spec.room_top(room.id) for room in rooms])
        return floor.slab_bottom - BAND_EPS_M, top - 0.002
    return -0.2, spec.wall_height - 0.002


def extract_floor_mesh(mesh: Mesh, spec: spec_mod.SceneSpec,
                       floor: spec_mod.FloorSpec) -> Mesh:
    """Triangles whose centroid lies in a storey's band and the building's XY.

    A centroid test (rather than "any vertex") keeps the storey boundary tight:
    the upper slab is a single box, its bottom face belongs to the lower storey
    only as a downward ceiling, and its top face is the upper storey's floor.
    Triangles that straddle the plane keep their natural side by their centroid.
    """
    if len(mesh) == 0:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), mesh.source)
    bounds_lo, bounds_hi = scene_bounds(spec)
    z_low, z_high = floor_band(spec, floor)
    tri = mesh.vertices[mesh.triangles]
    centroid = tri.mean(axis=1)
    finite = np.all(np.isfinite(centroid), axis=1)
    inside = np.all(centroid >= bounds_lo[None, :], axis=1) & \
        np.all(centroid <= bounds_hi[None, :], axis=1)
    band = (centroid[:, 2] >= z_low) & (centroid[:, 2] <= z_high)
    # degenerate slivers confuse Recast and carry no walkable area
    edge_a = tri[:, 1] - tri[:, 0]
    edge_b = tri[:, 2] - tri[:, 0]
    area2 = np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    keep = finite & inside & band & (area2 > 1e-10)
    return mesh.subset(np.nonzero(keep)[0])


def write_floor_obj(mesh: Mesh, path: str) -> dict:
    """Write a floor mesh as a Habitat Y-up OBJ (``(x, z, -y)``)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if len(mesh) == 0:
        raise ValueError(f"{mesh.source}: no triangles for {path}")
    points = np.stack([mesh.vertices[:, 0], mesh.vertices[:, 2],
                       -mesh.vertices[:, 1]], axis=1)
    bary = mesh.vertices[mesh.triangles].mean(axis=1)
    vertical = np.stack([bary[:, 0], bary[:, 2], -bary[:, 1]], axis=1)
    with open(path, "w", encoding="ascii") as handle:
        np.savetxt(handle, points, fmt="v %.6f %.6f %.6f")
        np.savetxt(handle, mesh.triangles + 1, fmt="f %d %d %d")
    return {
        "path": path,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "bounds_min": [float(v) for v in points.min(axis=0)],
        "bounds_max": [float(v) for v in points.max(axis=0)],
        "vertical_extent": [float(vertical[:, 1].min()), float(vertical[:, 1].max())],
        "bytes": os.path.getsize(path),
    }


# --------------------------------------------------------------------------- #
# source selection / fingerprints
# --------------------------------------------------------------------------- #
def source_candidates(entry: SceneEntry) -> List[str]:
    candidates: List[str] = []
    if entry.mesh_path:
        candidates.append(entry.mesh_path)
    candidates.extend([
        os.path.join(entry.build_dir, "furnished.ply"),
        os.path.join(entry.build_dir, "shell.ply"),
        os.path.join(entry.build_dir, "scene.glb"),
    ])
    return candidates


def resolve_mesh(entry: SceneEntry, source: Optional[str] = None) -> str:
    if source:
        path = source if os.path.isabs(source) else os.path.join(REPO_ROOT, source)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    for path in source_candidates(entry):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"{entry.scene_id}: no furnished.ply / shell.ply / scene.glb under {entry.build_dir}")


def read_mesh(path: str) -> Mesh:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".ply":
        return read_ply(path)
    if suffix == ".glb":
        return read_glb(path)
    raise ValueError(f"{path}: unsupported mesh format (use .ply or .glb)")


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def spec_sha256(spec: spec_mod.SceneSpec) -> str:
    payload = spec.raw if spec.raw is not None else {}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def scene_fingerprint(entry: SceneEntry, spec: spec_mod.SceneSpec,
                      mesh_path: str) -> str:
    """Cache key for the source mesh + spec (not per floor or profile)."""
    parts = [entry.scene_id, spec_sha256(spec), file_sha256(mesh_path)]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed from strings/numbers (no Python hash salt)."""
    blob = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little")
