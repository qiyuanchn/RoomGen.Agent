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
import re

try:  # inside Blender
    import bpy
    from mathutils import Matrix, Quaternion, Vector
except ImportError:  # host Python: the manifest helpers are still testable
    bpy = None
    Matrix = Quaternion = Vector = None

from blender_agent import spec as spec_mod

GLB_NAME = "scene.glb"
MANIFEST_NAME = "scene3d_manifest.json"
ENVIRONMENT_COLLECTION = "environment"
EYE_HEIGHT_M = 1.5

#: fields every exported semantic object carries (GLB ``node.extras`` + manifest)
SEMANTIC_FIELDS = ("instance_id", "category_id", "room_id", "floor_id")
SEMANTIC_SCHEMA_VERSION = 1

#: builtin props map to a canonical object class, so a ``builtin:dining_chair``
#: and a ``ph:dining_chair`` share the same navigation category
BUILTIN_CATEGORIES = {
    "rug": "rug",
    "table_lamp": "lamp",
    "kitchen_cabinet": "cabinet",
    "kitchen_upper": "cabinet",
    "kitchen_stove": "stove",
    "kitchen_sink": "sink",
    "kitchen_fridge": "fridge",
    "bath_toilet": "toilet",
    "bath_basin": "basin",
    "bath_shower": "shower",
    "study_desk": "desk",
    "study_chair": "chair",
    "dining_table": "table",
    "dining_chair": "chair",
    "wardrobe": "wardrobe",
    "entry_shoe_cabinet": "cabinet",
    "storage_shelf": "shelf",
    "laundry_washer": "washer",
}

#: object-class words found in asset ids / Poly Haven categories -> category id
CATEGORY_WORDS = {
    "bed": "bed", "beds": "bed",
    "sofa": "sofa", "sofas": "sofa", "couch": "sofa",
    "chair": "chair", "chairs": "chair", "armchair": "chair",
    "table": "table", "tables": "table", "desk": "desk", "desks": "desk",
    "nightstand": "nightstand", "nightstands": "nightstand",
    "cabinet": "cabinet", "cabinets": "cabinet",
    "wardrobe": "wardrobe", "wardrobes": "wardrobe",
    "shelf": "shelf", "shelves": "shelf", "bookshelf": "shelf", "bookshelves": "shelf",
    "rug": "rug", "rugs": "rug", "carpet": "rug",
    "lamp": "lamp", "lamps": "lamp", "light": "lamp", "lights": "lamp", "lighting": "lamp",
    "plant": "plant", "plants": "plant", "planter": "planter",
    "vase": "vase", "vases": "vase",
    "book": "book", "books": "book",
    "basket": "basket", "baskets": "basket",
    "frame": "picture_frame", "picture": "picture_frame", "pictures": "picture_frame",
    "painting": "picture_frame", "mirror": "mirror",
    "stove": "stove", "sink": "sink", "fridge": "fridge", "refrigerator": "fridge",
    "toilet": "toilet", "basin": "basin", "shower": "shower", "washer": "washer",
    "tv": "tv", "television": "tv", "monitor": "monitor",
    "ottoman": "ottoman", "bench": "bench", "stool": "stool",
}

#: labels that say what a category is *not* (too generic for a navigation goal)
GENERIC_CATEGORY_WORDS = {
    "furniture", "props", "decorative", "seating", "table", "tables", "shelves",
    "lighting", "containers", "plants", "potted", "nature", "wall", "decoration",
    "rigged", "office", "collection", "misc", "other",
}

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_NUMERIC_SUFFIX_RE = re.compile(r"[_-]\d+$")
_DUPLICATE_SUFFIX_RE = re.compile(r"^(?P<base>.*)\.(?P<suffix>\d{3})$")
_FIXTURE_PART_SUFFIXES = ("_base", "_stem", "_shade")


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
# semantic identity (instance / category / room / floor)
# --------------------------------------------------------------------------- #
def _slug(value) -> str:
    """Stable lowercase key: ``Modern Coffee Table`` -> ``modern_coffee_table``."""
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    return text.strip("_") or "unknown"


def strip_duplicate_suffix(name: str) -> str:
    """Split Blender's duplicate-name suffix: ``rug_x.001`` -> ``rug_x``."""
    match = _DUPLICATE_SUFFIX_RE.match(name)
    return match.group("base") if match else name


def duplicate_suffix_order(name: str) -> int:
    """Creation order Blender encoded in ``name``: no suffix = 0, ``.001`` = 1."""
    match = _DUPLICATE_SUFFIX_RE.match(name)
    return int(match.group("suffix")) if match else 0


def _asset_stem(asset_id: str) -> str:
    """Model id without its trailing variant number / dataset UUID."""
    stem = _UUID_RE.sub("", str(asset_id))
    stem = re.split(r"__+", stem)[0]
    stem = _NUMERIC_SUFFIX_RE.sub("", stem)
    return _slug(stem)


def _category_from_words(words) -> str:
    for word in words:
        category = CATEGORY_WORDS.get(_slug(word))
        if category:
            return category
    return ""


def asset_category(asset: str, asset_index: dict = None) -> str:
    """Canonical object class of a furniture asset.

    Builtin props use :data:`BUILTIN_CATEGORIES`; model ids first match a class
    word in the id itself (``sofa_02`` -> ``sofa``, ``potted_plant_01`` ->
    ``plant``), then the asset index's explicit ``category`` / category list
    (generated and 3D-FRONT entries), then the cleaned id stem.  The function is
    pure and deterministic: the same spec builds the same category ids.
    """
    scheme, _, asset_id = str(asset).partition(":")
    if scheme == "builtin":
        return BUILTIN_CATEGORIES.get(asset_id) or _slug(asset_id)
    if scheme == "front3d":
        return "front3d_furniture"

    entry = (asset_index or {}).get(f"{scheme}:{asset_id}")
    if entry:
        explicit = entry.get("category")
        if explicit:
            return _slug(explicit)

    category = _category_from_words(re.split(r"[^a-z0-9]+", asset_id.lower()))
    if category:
        return category
    if entry:
        categories = [value for value in (entry.get("categories") or ()) if value]
        category = _category_from_words(categories)
        if category:
            return category
        for value in categories:
            slug = _slug(value)
            if slug not in GENERIC_CATEGORY_WORDS and not slug.startswith("collection"):
                return slug
    return _asset_stem(asset_id)


def load_asset_index(root: str = None) -> dict:
    """Merge the polyhaven / midi3d / generated cache indexes for categories.

    Keyed ``scheme:asset_id`` (``ph:sofa_02``) so equal ids in different caches
    never collide.  A missing cache is fine: :func:`asset_category` falls back
    to the id itself.  ``root`` may be the assets directory or its
    ``polyhaven`` child (the run_build ``--cache`` value).
    """
    if root is None:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "third_party", "assets")
    root = os.path.abspath(os.path.expanduser(root))
    if os.path.basename(root) == "polyhaven":
        root = os.path.dirname(root)
    merged = {}
    for scheme, directory in (("ph", "polyhaven"), ("midi3d", "midi3d"), ("gen", "generated")):
        path = os.path.join(root, directory, "index.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, dict):
            continue
        for asset_id, entry in models.items():
            if isinstance(entry, dict):
                merged[f"{scheme}:{asset_id}"] = entry
    return merged


def _room_floor_id(spec, room_id: str) -> str:
    if spec is None or not room_id:
        return ""
    for room in getattr(spec, "rooms", []):
        if room.id == room_id:
            return str(getattr(room, "floor_id", "") or "")
    return ""


def _semantic_ids(instance_id: str, category_id: str, room_id: str, floor_id: str) -> dict:
    return {
        "instance_id": str(instance_id),
        "category_id": str(category_id),
        "room_id": str(room_id or ""),
        "floor_id": str(floor_id or ""),
    }


def _record_semantic(object_entries: dict, instances: dict, object_name: str, ids: dict,
                     **extra) -> bool:
    """Add one object -> four IDs mapping and aggregate it per instance."""
    if object_name in object_entries:
        return False
    entry = dict(ids)
    entry.update(extra)
    object_entries[object_name] = entry
    instance = instances.setdefault(ids["instance_id"], {
        **ids,
        "objects": [],
        "source": entry.get("source"),
    })
    for field in SEMANTIC_FIELDS:
        if instance.get(field) != ids[field]:
            raise RuntimeError(
                f"[export_glb] instance '{ids['instance_id']}' was assigned conflicting "
                f"{field}: '{instance.get(field)}' vs '{ids[field]}' (object '{object_name}')")
    for key in ("name", "asset", "kind", "placement_index", "to_room", "to_floor"):
        if key in entry and key not in instance:
            instance[key] = entry[key]
    if object_name not in instance["objects"]:
        instance["objects"].append(object_name)
    return True


def _furniture_kind(item) -> str:
    scheme, _, asset_id = str(item.asset).partition(":")
    if scheme == "front3d":
        return "front3d"
    if scheme == "builtin":
        if asset_id == "rug":
            return "rug"
        if asset_id == "table_lamp":
            return "lamp"
        return "fixture"
    return "model"


def _group_by_duplicate_suffix(names) -> list:
    """One group per placement: parts sharing Blender's duplicate suffix."""
    groups = {}
    for name in names:
        groups.setdefault(duplicate_suffix_order(name), []).append(name)
    return [sorted(groups[key]) for key in sorted(groups)]


def _furniture_object_groups(spec, objects) -> dict:
    """Map ``(kind, placement label)`` -> list of object groups in creation order.

    ``run_build.place_furniture`` names its output deterministically:
    ``furn_<label>`` (model instance empty), ``front3d_<label>``,
    ``rug_<label>``, ``lamp_<label>_{base,stem,shade}`` and
    ``fixture_<label>_<part>``.  Blender appends ``.001``/``.002`` to repeated
    names in creation order, so the duplicate suffix is the placement ordinal.
    """
    labels = [item.name or f"item_{index}" for index, item in enumerate(spec.furniture)]
    fixture_labels = sorted(
        {labels[i] for i, item in enumerate(spec.furniture) if _furniture_kind(item) == "fixture"},
        key=len, reverse=True)
    lamp_labels = {labels[i] for i, item in enumerate(spec.furniture)
                   if _furniture_kind(item) == "lamp"}
    buckets = {key: {} for key in ("model", "front3d", "rug", "lamp", "fixture")}

    for obj in objects:
        name = getattr(obj, "name", None)
        if not name:
            continue
        base = strip_duplicate_suffix(name)
        obj_type = getattr(obj, "type", "MESH")
        if obj_type == "EMPTY":
            if base.startswith("furn_"):
                buckets["model"].setdefault(base[len("furn_"):], []).append(name)
            elif base.startswith("front3d_"):
                buckets["front3d"].setdefault(base[len("front3d_"):], []).append(name)
            continue
        if obj_type != "MESH":
            continue
        if base.startswith("rug_"):
            buckets["rug"].setdefault(base[len("rug_"):], []).append(name)
        root = base
        for part in _FIXTURE_PART_SUFFIXES:
            if root.endswith(part):
                root = root[:-len(part)]
                break
        if root.startswith("lamp_"):
            candidate = root[len("lamp_"):]
            if candidate in lamp_labels:
                buckets["lamp"].setdefault(candidate, []).append(name)
        if base.startswith("fixture_"):
            rest = base[len("fixture_"):]
            for label in fixture_labels:
                if rest == label or rest.startswith(f"{label}_"):
                    buckets["fixture"].setdefault(label, []).append(name)
                    break

    return {kind: {label: _group_by_duplicate_suffix(names) for label, names in bucket.items()}
            for kind, bucket in buckets.items()}


def _furniture_semantics(spec, objects, asset_index, entries, instances) -> dict:
    """Assign every spec placement its stable instance/category/room/floor ids."""
    groups = _furniture_object_groups(spec, objects)
    consumed = {kind: {} for kind in groups}
    unbuilt = []
    for index, item in enumerate(spec.furniture):
        label = item.name or f"item_{index}"
        kind = _furniture_kind(item)
        ordinal = consumed[kind].get(label, 0)
        candidates = groups[kind].get(label) or []
        if ordinal >= len(candidates):
            # a skipped placement (e.g. a missing 3D-FRONT model) built no
            # object, so it must not consume the next placement's duplicate group
            unbuilt.append({"index": index, "name": label, "asset": str(item.asset),
                            "reason": "no matching scene object"})
            continue
        consumed[kind][label] = ordinal + 1
        names = candidates[ordinal]
        ids = _semantic_ids(
            f"furniture:{index:04d}",
            asset_category(item.asset, asset_index),
            item.room,
            _room_floor_id(spec, item.room),
        )
        for name in names:
            _record_semantic(entries, instances, name, ids,
                             source="furniture", placement_index=index,
                             name=label, asset=str(item.asset), kind=kind)
    return {"unbuilt": unbuilt}


def _shell_semantics(spec, object_names, entries, instances) -> None:
    """Floors, ceilings, walls, stairs, ramps and rails of a spec-built shell."""
    if spec is None:
        return

    for room in spec.rooms:
        floor_id = str(getattr(room, "floor_id", "") or "")
        name = f"floor_{room.id}"
        if name in object_names:
            _record_semantic(entries, instances, name,
                             _semantic_ids(f"floor:{room.id}", "floor", room.id, floor_id),
                             source="shell")
        parts = spec.ceiling_regions(room.id)
        for part_index, _polygon in enumerate(parts):
            name = (f"ceiling_{room.id}" if len(parts) == 1
                    else f"ceiling_{room.id}_{part_index}")
            if name in object_names:
                _record_semantic(entries, instances, name,
                                 _semantic_ids(f"ceiling:{room.id}", "ceiling", room.id, floor_id),
                                 source="shell")

    storeys = spec.floors or [None]
    for floor in storeys:
        rooms = spec.floor_rooms(floor.id) if floor is not None else list(spec.rooms)
        floor_id = floor.id if floor is not None else ""
        for index, segment in enumerate(spec_mod.build_wall_segments(rooms)):
            label = "_".join(segment.owners) or "wall"
            name = (f"wall_{index:02d}_{label}_"
                    f"{segment.p0[0]:.2f}_{segment.p0[1]:.2f}").replace("-", "m")
            if name not in object_names:
                continue
            owner = segment.owners[0] if segment.owners else ""
            entry_ids = _semantic_ids(
                f"wall:{floor_id or 'single'}:{index:02d}", "wall", owner,
                segment.floor_id or floor_id)
            _record_semantic(entries, instances, name, entry_ids, source="shell")
            entries[name]["rooms"] = list(segment.owners)

    for plan_index, plan in enumerate(spec.stair_plans()):
        connector = plan.connector
        label = f"{connector.from_room}_{connector.to_room}"
        ids = _semantic_ids(
            f"stair:{plan_index:02d}:{label}", "stair",
            connector.from_room, connector.from_floor)
        for box_index, box in enumerate(plan.boxes):
            name = f"stair_{label}_{box['role']}_{box_index:02d}"
            if name in object_names:
                _record_semantic(entries, instances, name, ids, source="shell",
                                 to_room=connector.to_room, to_floor=connector.to_floor)

    for plan_index, plan in enumerate(spec.ramp_plans()):
        connector = plan.connector
        label = f"{connector.from_room}_{connector.to_room}"
        name = f"ramp_{label}"
        if name not in object_names:
            continue
        ids = _semantic_ids(
            f"ramp:{plan_index:02d}:{label}", "ramp",
            connector.from_room, connector.from_floor)
        _record_semantic(entries, instances, name, ids, source="shell",
                         to_room=connector.to_room, to_floor=connector.to_floor)

    for entry in spec.railing_boxes():
        name = entry["name"]
        if name not in object_names:
            continue
        void = str(entry.get("void") or "void")
        floor_id = str(entry.get("floor") or "")
        remainder = name[len(f"railing_{void}_"):] if name.startswith(f"railing_{void}_") else name
        span = remainder.split("_", 1)[0] or "0"
        room_id = ""
        polygon = entry.get("polygon") or []
        if polygon:
            cx = sum(float(point[0]) for point in polygon) / len(polygon)
            cy = sum(float(point[1]) for point in polygon) / len(polygon)
            for room in spec.floor_rooms(floor_id) if floor_id else spec.rooms:
                if spec_mod.point_in_polygon((cx, cy), room.polygon):
                    room_id = room.id
                    break
        _record_semantic(
            entries, instances, name,
            _semantic_ids(f"railing:{floor_id or 'single'}:{void}:{span}",
                          "railing", room_id, floor_id),
            source="shell", void=void)


def _detail_semantics(spec, object_names, entries, instances) -> None:
    """Skirting, door/window casings, sills and glass (P0/P1 details layer)."""
    if spec is None:
        return

    rooms = sorted(spec.rooms, key=lambda room: len(room.id), reverse=True)
    for name in sorted(object_names):
        if not name.startswith("skirting_"):
            continue
        for room in rooms:
            if name.startswith(f"skirting_{room.id}_"):
                _record_semantic(
                    entries, instances, name,
                    _semantic_ids(f"skirting:{room.id}", "skirting",
                                  room.id, getattr(room, "floor_id", "") or ""),
                    source="detail")
                break

    for index, opening in enumerate(spec.openings):
        room_id = opening.between[0] if opening.between else ""
        prefix = f"{opening.kind}_casing_{index}_"
        ids = _semantic_ids(f"detail:opening:{index}", f"{opening.kind}_casing",
                            room_id, _room_floor_id(spec, room_id))
        for name in sorted(object_names):
            if name.startswith(prefix):
                _record_semantic(entries, instances, name, ids, source="detail")

    for index, window in enumerate(spec.windows):
        prefixes = (
            (f"window_casing_{index}_", "window_casing"),
            (f"window_sill_{index}", "window_sill"),
            (f"window_glass_{index}", "window_glass"),
        )
        for prefix, category in prefixes:
            ids = _semantic_ids(f"detail:window:{index}:{category}", category, window.room,
                                _room_floor_id(spec, window.room))
            for name in sorted(object_names):
                if name.startswith(prefix):
                    _record_semantic(entries, instances, name, ids, source="detail")


def _fallback_semantic(name, objects_by_name, entries, instances) -> None:
    """Every exported object still gets four ids, even if no spec rule matched."""
    obj = objects_by_name.get(name)
    custom = getattr(obj, "get", None)
    front3d_type = custom("front3d_type") if callable(custom) else None
    if front3d_type:
        category = _slug(front3d_type)
        source = "front3d_shell"
    elif name == "sky_dome":
        category = "sky_dome"
        source = "environment"
    else:
        category = _slug(strip_duplicate_suffix(name).split("_", 1)[0])
        source = "fallback"
    _record_semantic(entries, instances, name,
                     _semantic_ids(f"object:{name}", category, "", ""),
                     source=source)


def build_semantics(spec, objects, asset_index: dict = None) -> dict:
    """Build the object-name -> four-IDs table for every exported scene object.

    ``objects`` is any iterable of Blender-like objects (``.name`` / ``.type``);
    no bpy API is touched, so the table is unit-testable on the host.  Furniture
    ids follow ``spec.furniture`` order (``furniture:0007``), shell objects use
    their architectural owner (``floor:living``, ``wall:f0:03``), and repeated
    pieces of one placement (a fixture's parts, a lamp's three meshes) share the
    placement's instance id.

    Returns the ``semantics`` block written to ``scene3d_manifest.json``:
    ``objects`` (the mapping table), ``instances`` (one row per semantic unit),
    ``categories`` (counts), and the ids of placements that were never built.
    """
    objects = list(objects)
    objects_by_name = {getattr(obj, "name", None): obj for obj in objects
                       if getattr(obj, "name", None)}
    object_names = set(objects_by_name)
    entries, instances = {}, {}

    if spec is not None:
        furniture_report = _furniture_semantics(spec, objects, asset_index, entries, instances)
        _shell_semantics(spec, object_names, entries, instances)
        _detail_semantics(spec, object_names, entries, instances)
    else:
        furniture_report = {"unbuilt": []}

    for name in sorted(object_names):
        if name not in entries:
            _fallback_semantic(name, objects_by_name, entries, instances)

    categories = {}
    for instance in instances.values():
        category = instance["category_id"]
        record = categories.setdefault(category, {"instance_count": 0, "object_count": 0})
        record["instance_count"] += 1
        record["object_count"] += len(instance["objects"])

    fallback = sorted(name for name, entry in entries.items()
                      if entry.get("source") in ("fallback", "front3d_shell"))
    return {
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "id_fields": list(SEMANTIC_FIELDS),
        "categories": categories,
        "instances": instances,
        "objects": entries,
        "unbuilt_furniture": furniture_report["unbuilt"],
        "fallback_objects": fallback,
    }


def attach_glb_nodes(document: dict, semantics: dict) -> dict:
    """Record which GLB node indices belong to each semantic object.

    Collection instances are exported as an empty parent node carrying the
    extras plus child mesh nodes that do not (the child mesh is shared by every
    instance).  A child therefore inherits the nearest ancestor's ids; the
    manifest lists both parent and children under the placement so consumers do
    not have to know the hierarchy.
    """
    nodes = document.get("nodes") or []
    parents = {}
    for index, node in enumerate(nodes):
        for child in node.get("children") or ():
            parents[child] = index

    def resolved(node_index: int):
        current = node_index
        while current is not None:
            extras = nodes[current].get("extras") or {}
            if all(field in extras for field in SEMANTIC_FIELDS):
                return {field: str(extras[field]) for field in SEMANTIC_FIELDS}
            current = parents.get(current)
        return None

    def chain_names(node_index: int) -> list:
        names, current = [], node_index
        while current is not None:
            names.append(nodes[current].get("name"))
            current = parents.get(current)
        return names

    object_entries = semantics.get("objects") or {}
    for name, entry in object_entries.items():
        ids = {field: str(entry.get(field) or "") for field in SEMANTIC_FIELDS}
        hits = [index for index in range(len(nodes))
                if resolved(index) == ids and name in chain_names(index)]
        entry["glb_nodes"] = sorted(set(hits))

    for instance in (semantics.get("instances") or {}).values():
        nodes_for_instance = []
        for name in instance.get("objects") or ():
            nodes_for_instance.extend(object_entries.get(name, {}).get("glb_nodes") or ())
        instance["glb_nodes"] = sorted(set(nodes_for_instance))

    own_extras = set()
    for index, node in enumerate(nodes):
        extras = node.get("extras") or {}
        if all(field in extras for field in SEMANTIC_FIELDS):
            own_extras.add(index)
    node_table = {}
    for index in range(len(nodes)):
        ids = resolved(index)
        if ids is None:
            continue
        node_table[str(index)] = {
            "name": nodes[index].get("name"),
            "source": "self" if index in own_extras else "ancestor",
            **ids,
        }
    semantics["nodes"] = node_table
    return semantics


def verify_glb_semantics(document: dict, manifest: dict) -> tuple:
    """Check GLB node extras against the manifest's semantics table.

    Returns ``(failures, warnings, report)``.  A legacy manifest without a
    ``semantics`` block is not an error: the caller only gets a warning, so an
    older ``scene.glb`` still passes ``tools/check_scene_glb.py``.
    """
    failures, warnings = [], []
    semantics = (manifest or {}).get("semantics")
    if not semantics:
        return failures, ["no semantics table (legacy GLB): extras not checked"], {}

    nodes = document.get("nodes") or []
    object_entries = semantics.get("objects") or {}
    instance_entries = semantics.get("instances") or {}
    if not object_entries:
        failures.append("semantics table has no 'objects' mapping")
    if not instance_entries:
        failures.append("semantics table has no 'instances' mapping")

    parents = {}
    for index, node in enumerate(nodes):
        for child in node.get("children") or ():
            parents[child] = index

    own_extras = {}
    for index, node in enumerate(nodes):
        extras = node.get("extras")
        if not isinstance(extras, dict):
            continue
        present = [field for field in SEMANTIC_FIELDS if field in extras]
        if not present:
            continue
        if len(present) != len(SEMANTIC_FIELDS):
            failures.append(f"node {index} ({node.get('name')}) carries only {present}; "
                            f"expected all of {list(SEMANTIC_FIELDS)}")
            continue
        own_extras[index] = {field: str(extras[field]) for field in SEMANTIC_FIELDS}

    resolved = {}

    def resolve(index: int):
        if index in resolved:
            return resolved[index]
        current, found = index, None
        while current is not None:
            if current in own_extras:
                found = own_extras[current]
                break
            current = parents.get(current)
        resolved[index] = found
        return found

    def check_instance(ids: dict, where: str) -> None:
        instance = instance_entries.get(ids["instance_id"])
        if instance is None:
            failures.append(f"{where}: unknown instance_id '{ids['instance_id']}'")
            return
        for field in SEMANTIC_FIELDS:
            if str(instance.get(field) or "") != ids[field]:
                failures.append(
                    f"{where}: {field} '{ids[field]}' differs from manifest instance "
                    f"'{instance.get(field)}'")

    mesh_nodes = 0
    resolved_mesh_nodes = 0
    for index, node in enumerate(nodes):
        if "mesh" not in node:
            if index in own_extras:
                check_instance(own_extras[index], f"node {index} ({node.get('name')})")
            continue
        mesh_nodes += 1
        ids = resolve(index)
        if ids is None:
            failures.append(f"mesh node {index} ({node.get('name')}) has no instance/category/"
                            f"room/floor extras on itself or an ancestor")
            continue
        resolved_mesh_nodes += 1
        check_instance(ids, f"mesh node {index} ({node.get('name')})")

    listed_nodes = set()
    has_node_lists = any("glb_nodes" in entry for entry in object_entries.values())
    for name, entry in object_entries.items():
        missing = [field for field in SEMANTIC_FIELDS if field not in entry]
        if missing:
            failures.append(f"object '{name}' is missing {missing}")
            continue
        if not entry.get("category_id"):
            failures.append(f"object '{name}' has an empty category_id")
        ids = {field: str(entry.get(field) or "") for field in SEMANTIC_FIELDS}
        check_instance(ids, f"object '{name}'")
        if "glb_nodes" not in entry:
            warnings.append(f"object '{name}' has no glb_nodes (manifest without a GLB)")
            continue
        glb_nodes = entry.get("glb_nodes") or []
        if not glb_nodes:
            failures.append(f"object '{name}' maps to no GLB node")
        for index in glb_nodes:
            listed_nodes.add(index)
            if not (0 <= index < len(nodes)):
                failures.append(f"object '{name}' references GLB node {index}, file has {len(nodes)}")
                continue
            node_ids = resolve(index)
            if node_ids != ids:
                failures.append(f"object '{name}' / node {index} extras {node_ids} differ from "
                                f"the manifest {ids}")

    if has_node_lists:
        for index in own_extras:
            if index not in listed_nodes:
                failures.append(f"node {index} ({nodes[index].get('name')}) has extras but no "
                                f"manifest object lists it")
        for index, node in enumerate(nodes):
            if "mesh" in node and index not in listed_nodes:
                failures.append(f"mesh node {index} ({node.get('name')}) is not listed by any "
                                f"manifest object")

    node_table = semantics.get("nodes")
    if isinstance(node_table, dict):
        for key, entry in node_table.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                failures.append(f"semantics.nodes key '{key}' is not a GLB node index")
                continue
            if not (0 <= index < len(nodes)):
                failures.append(f"semantics.nodes references GLB node {index}, file has {len(nodes)}")
                continue
            ids = {field: str(entry.get(field) or "") for field in SEMANTIC_FIELDS}
            if resolve(index) != ids:
                failures.append(f"semantics.nodes[{index}] {ids} differs from the node extras "
                                f"{resolve(index)}")
            check_instance(ids, f"semantics.nodes[{index}]")
    elif has_node_lists:
        warnings.append("semantics has glb_nodes but no per-node table")

    floor_ids = {str(floor.get("id")) for floor in (manifest.get("floors") or [])}
    room_floor = {str(room.get("id")): (str(room.get("floor_id")) if room.get("floor_id") else "")
                  for room in (manifest.get("rooms") or [])}
    for name, entry in object_entries.items():
        room_id = str(entry.get("room_id") or "")
        floor_id = str(entry.get("floor_id") or "")
        if room_id and room_id in room_floor:
            if floor_id != room_floor[room_id]:
                failures.append(f"object '{name}' (room '{room_id}') has floor_id '{floor_id}', "
                                f"the manifest room says '{room_floor[room_id]}'")
        elif floor_id and floor_ids and floor_id not in floor_ids:
            failures.append(f"object '{name}' has unknown floor_id '{floor_id}'")

    for instance_id, instance in instance_entries.items():
        if not instance.get("objects"):
            failures.append(f"instance '{instance_id}' has no objects")
        for name in instance.get("objects") or ():
            entry = object_entries.get(name)
            if entry is None:
                failures.append(f"instance '{instance_id}' lists unknown object '{name}'")
            elif entry.get("instance_id") != instance_id:
                failures.append(f"instance '{instance_id}' lists object '{name}' with "
                                f"instance_id '{entry.get('instance_id')}'")

    categories = {}
    for instance_id, instance in instance_entries.items():
        categories.setdefault(instance.get("category_id"), []).append(instance_id)
    for category, ids in categories.items():
        if len(ids) != len(set(ids)):
            failures.append(f"category '{category}' repeats an instance id")

    report = {
        "instances": len(instance_entries),
        "objects": len(object_entries),
        "categories": len(categories),
        "nodes_with_extras": len(own_extras),
        "mesh_nodes": mesh_nodes,
        "resolved_mesh_nodes": resolved_mesh_nodes,
        "listed_nodes": len(listed_nodes),
    }
    return failures, warnings, report


# --------------------------------------------------------------------------- #
# viewer manifest
# --------------------------------------------------------------------------- #
def viewer_manifest(spec, glb_name: str, stats: dict, bounds: dict, skipped: list,
                    semantics: dict = None) -> dict:
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

    manifest = {
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
    if semantics is not None:
        manifest["semantics"] = semantics
    return manifest


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
                     include_environment: bool = False, asset_index: dict = None) -> dict:
    """Export the dressed scene to ``scene.glb`` plus the viewer manifest.

    Returns the manifest dict.  ``build_dir`` supplies the default ``scene.glb``
    location; ``scene3d_manifest.json`` always lands next to the GLB it belongs to.
    Every exported object carries ``instance_id`` / ``category_id`` /
    ``room_id`` / ``floor_id`` as glTF ``node.extras``; the same four ids form
    the manifest's ``semantics.objects`` mapping table.

    ``asset_index`` may carry the polyhaven/midi3d/generated cache entries for
    category lookup; when omitted the repository caches are read best-effort.
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

    if asset_index is None:
        asset_index = load_asset_index()
    selected_objects = [bpy.data.objects[name] for name in selected
                        if bpy.data.objects.get(name) is not None]
    semantics = build_semantics(spec, selected_objects, asset_index=asset_index)
    for name, entry in semantics["objects"].items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        for field in SEMANTIC_FIELDS:
            obj[field] = entry[field]

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
        export_extras=True,
        export_texture_dir="",
    )
    bpy.ops.export_scene.gltf(**kwargs)

    if not os.path.isfile(filepath):
        raise RuntimeError(f"[export_glb] the exporter did not write {filepath}")

    document = read_glb_document(filepath)
    stats = glb_file_stats(document, filepath)
    stats["blender_objects"] = len(selected)
    bounds = glb_bounds(document)
    attach_glb_nodes(document, semantics)
    manifest = viewer_manifest(spec, os.path.basename(filepath), stats, bounds, skipped,
                               semantics=semantics)
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
    print(f"[export_glb] semantics: {len(semantics['instances'])} instances, "
          f"{len(semantics['objects'])} objects, {len(semantics['categories'])} categories, "
          f"{len(semantics['fallback_objects'])} fallback objects")
    print(f"[export_glb] wrote {filepath}")
    print(f"[export_glb] wrote {manifest_path}")
    return manifest
