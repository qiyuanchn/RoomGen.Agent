"""GLB export for the browser 3D viewer (Blender side, used by ``run_build.py``).

``run_build.py --export-glb`` calls :func:`export_scene_glb` once the scene is
fully dressed (shell, details, furniture, materials) and writes:

``<build>/scene.glb``             the whole room, glTF binary, +Y up (glTF
                                  convention), textures embedded, no cameras
                                  and no lights (the viewer brings its own)
``<build>/scene3d_manifest.json`` cameras / rooms / bounds in glTF coordinates,
                                  i.e. exactly the numbers the web viewer needs
                                  to place its Orbit/Fly/Walk camera, plus the
                                  export statistics

Only ``bpy``/``mathutils`` (both inside Blender) plus the standard library are
used at build time.  The pure manifest helpers (:func:`blender_to_gltf` and
:func:`viewer_manifest`) also import on a host Python for ``tests/``: the bpy and
mathutils imports below are optional, and nothing outside the GLB-export path
touches them.

Coordinate conversion: the Blender scene is Z-up metres, glTF is Y-up metres,
so ``p_gltf = (x, z, -y)`` - the same mapping ``bpy.ops.export_scene.gltf``
applies with ``export_yup``.
"""

from __future__ import annotations

import datetime
import json
import math
import os

try:  # inside Blender
    import bpy
    from mathutils import Matrix, Quaternion, Vector
except ImportError:  # host Python: the manifest helpers are still testable
    bpy = None
    Matrix = Quaternion = Vector = None

GLB_NAME = "scene.glb"
MANIFEST_NAME = "scene3d_manifest.json"
ENVIRONMENT_COLLECTION = "environment"
EYE_HEIGHT_M = 1.5


# --------------------------------------------------------------------------- #
# coordinate helpers
# --------------------------------------------------------------------------- #
def blender_to_gltf(point) -> list:
    """Blender (X, Y, Z) metres -> glTF (X, Y, Z) metres."""
    return [float(point[0]), float(point[2]), -float(point[1])]


def blender_heading_to_gltf(yaw_deg: float) -> list:
    """Blender yaw (0 = +X, CCW in the XY plane) -> glTF unit forward vector."""
    yaw = math.radians(yaw_deg)
    return [math.cos(yaw), 0.0, -math.sin(yaw)]


# --------------------------------------------------------------------------- #
# object selection
# --------------------------------------------------------------------------- #
def _environment_objects() -> set:
    collection = bpy.data.collections.get(ENVIRONMENT_COLLECTION)
    if collection is None:
        return set()
    return {obj.name for obj in collection.all_objects}


def select_export_objects(include_environment: bool = False) -> tuple:
    """Select every mesh / collection instance, minus the render-only dome.

    Returns ``(selected_names, skipped)``.  The sky dome is a render depth trick
    rather than scene content (a >= 50 m emissive sphere that would dwarf the
    room in the viewer), so it stays out unless asked for.
    """
    environment = _environment_objects()
    selected, skipped = [], []
    for obj in bpy.context.scene.objects:
        if obj.type in ("CAMERA", "LIGHT"):
            skipped.append({"name": obj.name, "reason": obj.type.lower()})
            continue
        if obj.name in environment and not include_environment:
            skipped.append({"name": obj.name, "reason": "render-only environment (sky dome)"})
            continue
        if obj.type == "MESH" or (obj.type == "EMPTY" and obj.instance_collection is not None):
            selected.append(obj.name)
            continue
        skipped.append({"name": obj.name, "reason": f"unsupported type '{obj.type.lower()}'"})

    wanted = set(selected)
    for obj in bpy.context.scene.objects:
        obj.select_set(obj.name in wanted)
    return selected, skipped


def read_glb_document(path: str) -> dict:
    """The GLB's JSON chunk (header + chunk walk, no glTF library needed)."""
    import struct

    with open(path, "rb") as handle:
        magic, _version, length = struct.unpack("<4sII", handle.read(12))
        if magic != b"glTF":
            raise RuntimeError(f"[export_glb] {path} is not a binary glTF file")
        document = None
        while handle.tell() < length:
            chunk_length, chunk_type = struct.unpack("<II", handle.read(8))
            chunk = handle.read(chunk_length)
            if chunk_type == 0x4E4F534A:  # 'JSON'
                document = json.loads(chunk.decode("utf-8"))
    if document is None:
        raise RuntimeError(f"[export_glb] {path} has no JSON chunk")
    return document


def glb_file_stats(document: dict, path: str) -> dict:
    """Stats read back from the written GLB, so the manifest describes the file."""
    accessors = document.get("accessors", [])
    meshes = document.get("meshes", [])
    per_mesh = []
    for mesh in meshes:
        triangles = 0
        for primitive in mesh.get("primitives", []):
            if "indices" in primitive:
                triangles += accessors[primitive["indices"]]["count"] // 3
            else:
                triangles += accessors[primitive["attributes"]["POSITION"]]["count"] // 3
        per_mesh.append(triangles)

    nodes = document.get("nodes", [])
    return {
        "meshes": len(meshes),
        "nodes": len(nodes),
        "mesh_nodes": sum(1 for node in nodes if "mesh" in node),
        "triangles_unique": sum(per_mesh),
        "triangles_rendered": sum(per_mesh[node["mesh"]] for node in nodes if "mesh" in node),
        "materials": len(document.get("materials", [])),
        "textures": len(document.get("textures", [])),
        "images": len(document.get("images", [])),
        "bytes": os.path.getsize(path),
    }


def _node_matrix(node: dict) -> Matrix:
    if "matrix" in node:  # glTF matrices are column-major
        m = node["matrix"]
        return Matrix((
            (m[0], m[4], m[8], m[12]),
            (m[1], m[5], m[9], m[13]),
            (m[2], m[6], m[10], m[14]),
            (m[3], m[7], m[11], m[15]),
        ))
    translation = Vector(node.get("translation", (0.0, 0.0, 0.0)))
    rotation = node.get("rotation", (0.0, 0.0, 0.0, 1.0))  # glTF order: x, y, z, w
    scale = Vector(node.get("scale", (1.0, 1.0, 1.0)))
    quaternion = Quaternion((rotation[3], rotation[0], rotation[1], rotation[2]))
    return (Matrix.Translation(translation)
            @ quaternion.to_matrix().to_4x4()
            @ Matrix.Diagonal(scale.to_4d()))


def glb_bounds(document: dict) -> dict:
    """World-space AABB of the written GLB, from accessor min/max + node matrices.

    Read from the file on purpose: the numbers describe what the viewer loads,
    and the depsgraph hand-outs (``object_instances`` matrices) are a reused
    buffer whose values proved unreliable straight after the settle bake.
    """
    accessors = document.get("accessors", [])
    meshes = document.get("meshes", [])
    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    roots = scenes[document.get("scene", 0)].get("nodes", []) if scenes else []

    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3

    def walk(index: int, parent: Matrix) -> None:
        node = nodes[index]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            for primitive in meshes[node["mesh"]].get("primitives", []):
                accessor = accessors[primitive["attributes"]["POSITION"]]
                minimum, maximum = accessor.get("min"), accessor.get("max")
                if not minimum or not maximum:
                    continue
                for corner in range(8):
                    point = Vector((minimum[0] if corner & 1 else maximum[0],
                                    minimum[1] if corner & 2 else maximum[1],
                                    minimum[2] if corner & 4 else maximum[2]))
                    world_point = world @ point
                    for axis in range(3):
                        lo[axis] = min(lo[axis], world_point[axis])
                        hi[axis] = max(hi[axis], world_point[axis])
        for child in node.get("children", []):
            walk(child, world)

    for root in roots:
        walk(root, Matrix.Identity(4))
    if lo[0] == float("inf"):
        return None
    return {"min": lo, "max": hi}


# --------------------------------------------------------------------------- #
# viewer manifest
# --------------------------------------------------------------------------- #
def viewer_manifest(spec, glb_name: str, stats: dict, bounds: dict, skipped: list) -> dict:
    """Cameras + rooms + floors + bounds in glTF coordinates for the web viewer.

    Every room and camera carries the storey it belongs to (``floor_id``,
    ``floor_elevation`` in Blender metres).  A room centre is placed at
    ``floor elevation + eye height`` in glTF Y, so the viewer can jump to a
    second-storey room without guessing from a global constant; camera poses
    keep their authored world Z (already elevation + eye height).
    """
    floor_by_room = {}
    floors = []
    if spec is not None:
        for floor in getattr(spec, "floors", []):
            floors.append({
                "id": floor.id,
                "elevation": float(floor.elevation),
                "height": float(floor.height),
                "slab_thickness": float(floor.slab_thickness),
                "slab_bottom": float(floor.slab_bottom),
                "top": float(floor.top),
                "rooms": [room.id for room in spec.rooms if room.floor_id == floor.id],
            })
            for room in spec.rooms:
                if room.floor_id == floor.id:
                    floor_by_room[room.id] = floor

    cameras = []
    if spec is not None:
        for camera in spec.cameras:
            yaw_deg = float(getattr(camera, "yaw", 0.0) or 0.0)
            floor = floor_by_room.get(camera.room)
            cameras.append({
                "name": camera.name,
                "room": getattr(camera, "room", None),
                "floor_id": floor.id if floor is not None else None,
                "floor_elevation": float(floor.elevation) if floor is not None else 0.0,
                "position": blender_to_gltf(camera.pos),
                "yaw_deg": yaw_deg,
                "forward": blender_heading_to_gltf(yaw_deg),
            })

    rooms = []
    if spec is not None:
        first_camera = {}
        for camera in cameras:
            first_camera.setdefault(camera.get("room"), camera["name"])
        for room in spec.rooms:
            xs = [float(p[0]) for p in room.polygon]
            ys = [float(p[1]) for p in room.polygon]
            floor = floor_by_room.get(room.id)
            elevation = float(floor.elevation) if floor is not None else 0.0
            height = float(spec.room_height(room.id))
            center = blender_to_gltf(((min(xs) + max(xs)) / 2.0,
                                      (min(ys) + max(ys)) / 2.0,
                                      elevation + EYE_HEIGHT_M))
            rooms.append({
                "id": room.id,
                "type": room.type,
                "camera": first_camera.get(room.id),
                "floor_id": floor.id if floor is not None else room.floor_id,
                "floor_elevation": elevation,
                "floor_height": height,
                "center": center,
                "size": [max(xs) - min(xs), max(ys) - min(ys), height],
                "polygon": [[float(p[0]), -float(p[1])] for p in room.polygon],
            })

    return {
        "kind": "roomgen_scene3d",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "blender_version": getattr(getattr(bpy, "app", None), "version_string", None),
        "glb": glb_name,
        "up_axis": "Y",
        "units": "meter",
        "eye_height": EYE_HEIGHT_M,
        "bounds": bounds,
        "floors": floors,
        "cameras": cameras,
        "rooms": rooms,
        "stats": stats,
        "excluded": skipped,
    }


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def _gltf_export_kwargs(**wanted) -> dict:
    """Drop kwargs the installed glTF exporter does not know (version tolerant)."""
    try:
        properties = bpy.ops.export_scene.gltf.get_rna_type().properties
    except Exception:  # pragma: no cover - RNA introspection should not fail
        return wanted
    supported = {name for name, _ in properties.items()}
    return {key: value for key, value in wanted.items() if key in supported}


def export_scene_glb(build_dir: str, spec=None, *, filepath: str = None,
                     include_environment: bool = False) -> dict:
    """Export the dressed scene to ``scene.glb`` plus the viewer manifest.

    Returns the manifest dict.  ``build_dir`` supplies the default ``scene.glb``
    location; ``scene3d_manifest.json`` always lands next to the GLB it belongs to.
    """
    build_dir = os.path.abspath(build_dir)
    filepath = os.path.abspath(filepath) if filepath else os.path.join(build_dir, GLB_NAME)
    out_dir = os.path.dirname(filepath)
    os.makedirs(out_dir, exist_ok=True)

    selected, skipped = select_export_objects(include_environment=include_environment)
    if not selected:
        raise RuntimeError("[export_glb] nothing to export: the scene has no mesh objects")

    # The settle step bakes transforms straight onto the objects; without a view
    # layer update the exporter can still see the pre-settle poses.
    bpy.context.view_layer.update()

    kwargs = _gltf_export_kwargs(
        filepath=filepath,
        export_format="GLB",
        use_selection=True,
        export_apply=True,          # the details layer keeps live bevel modifiers
        export_yup=True,            # glTF convention: +Y up
        export_cameras=False,
        export_lights=False,
        export_animations=False,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_texcoords=True,
        export_normals=True,
        export_tangents=False,
        export_extras=False,
        export_texture_dir="",
    )
    bpy.ops.export_scene.gltf(**kwargs)

    if not os.path.isfile(filepath):
        raise RuntimeError(f"[export_glb] the exporter did not write {filepath}")

    document = read_glb_document(filepath)
    stats = glb_file_stats(document, filepath)
    stats["blender_objects"] = len(selected)
    bounds = glb_bounds(document)
    manifest = viewer_manifest(spec, os.path.basename(filepath), stats, bounds, skipped)
    manifest["glb_path"] = filepath
    manifest["include_environment"] = bool(include_environment)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(
        f"[export_glb] {os.path.basename(filepath)}: {stats['bytes'] / 1e6:.1f} MB, "
        f"{stats['meshes']} meshes in {stats['mesh_nodes']} nodes, "
        f"{stats['triangles_rendered']} triangles, {stats['materials']} materials, "
        f"{stats['images']} images, {len(skipped)} objects excluded"
    )
    if bounds is not None:
        size = [bounds["max"][i] - bounds["min"][i] for i in range(3)]
        print(f"[export_glb] bounds (glTF metres): {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}, "
              f"min {[round(v, 2) for v in bounds['min']]}")
    print(f"[export_glb] wrote {filepath}")
    print(f"[export_glb] wrote {manifest_path}")
    return manifest
