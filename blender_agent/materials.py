"""Poly Haven PBR texture sets -> Principled BSDF, with physically-scaled box mapping.

The shell prisms produced by ``shell.py`` carry no UVs and span arbitrary
polygons, so textures are applied with Blender's *box projection*: an Object
coordinate (metres, because the shell objects have identity transforms) is
scaled by ``1 / tile_size`` from the Poly Haven metadata, which keeps plank and
plaster scale physically correct whatever the wall's shape.

Map wiring (``arm`` is the packed AO/Roughness/Metalness map; using it instead
of a standalone roughness jpg saves one 2K texture per material):

    Diffuse (sRGB)  -> Base Color
    arm.G           -> Roughness
    arm.B           -> Metallic
    nor_gl          -> Normal Map (OpenGL convention, same as Blender)

Spec hook (``materials`` in the scene spec, all optional)::

    "materials": {"floor": "ph:oak_wood_planks", "wall": "ph:beige_wall_001",
                  "rug": "ph:wool_boucle", "bedding": "ph:rough_linen",
                  "floor_tile_m": 1.2, "wall_tile_m": 3.0}

Any field left out falls back to the P0 flat Principled colours, so a P0 spec
keeps working unchanged.
"""

from __future__ import annotations

import os

from blender_agent import assets as assets_mod


def _bpy():
    import bpy

    return bpy


def _socket(bsdf, *names):
    for name in names:
        if name in bsdf.inputs:
            return bsdf.inputs[name]
    raise KeyError(f"none of {names} on {bsdf.name}")


def load_image(cache_dir: str, rel_path: str, non_color: bool = False):
    """Load (or reuse) a cached image, with the right colour space."""
    bpy = _bpy()
    path = os.path.join(cache_dir, rel_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"texture file missing: {path} (run assets.py fetch)")
    image = bpy.data.images.get(os.path.basename(path))
    if image is None:
        image = bpy.data.images.load(path)
    image.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return image


def texture_material(cache_dir: str, texture_id: str, name: str = None, tile_m: float = None,
                     tile_scale: float = 1.0, roughness_factor: float = 1.0, metallic: float = 0.0,
                     normal_strength: float = 1.0, tint=None, tint_mix: float = 0.5):
    """Build a Principled material from one cached Poly Haven texture set."""
    bpy = _bpy()
    index = assets_mod.load_index(cache_dir)
    kind, entry = assets_mod.asset_entry(index, texture_id)
    if kind != "texture":
        raise ValueError(f"'{texture_id}' is a model, not a texture")
    tile = tile_m if tile_m else max(entry["tile_size_m"]) * tile_scale
    name = name or f"mat_{texture_id}"

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    coords = tree.nodes.new("ShaderNodeTexCoord")
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0 / tile, 1.0 / tile, 1.0 / tile)
    tree.links.new(coords.outputs["Object"], mapping.inputs["Vector"])

    def box_texture(rel_path: str, non_color: bool):
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = load_image(cache_dir, rel_path, non_color=non_color)
        node.projection = "BOX"
        node.projection_blend = 0.35
        node.extension = "REPEAT"
        node.interpolation = "Smart"
        tree.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    maps = entry["maps"]
    diffuse = box_texture(maps["Diffuse"], non_color=False)
    base_color = diffuse.outputs["Color"]
    if tint is not None:
        # mix towards a flat colour: mutes busy patterns (rug) or warms a surface
        mix = tree.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MIX"
        mix.inputs["Fac"].default_value = float(tint_mix)
        mix.inputs["Color2"].default_value = (tint[0], tint[1], tint[2], 1.0)
        tree.links.new(base_color, mix.inputs["Color1"])
        base_color = mix.outputs["Color"]
    tree.links.new(base_color, _socket(bsdf, "Base Color"))

    packed = maps.get("arm") or maps.get("Rough")
    if packed:
        arm = box_texture(packed, non_color=True)
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.mode = "RGB"
        tree.links.new(arm.outputs["Color"], separate.inputs["Color"])
        if maps.get("arm"):
            roughness = separate.outputs["Green"]  # arm: R=AO, G=Roughness, B=Metal
            metal = separate.outputs["Blue"]
            if metallic == 0.0:
                tree.links.new(metal, _socket(bsdf, "Metallic"))
            else:
                _socket(bsdf, "Metallic").default_value = metallic
        else:
            roughness = separate.outputs["Red"]
        if roughness_factor != 1.0:
            scale = tree.nodes.new("ShaderNodeMath")
            scale.operation = "MULTIPLY"
            scale.inputs[1].default_value = roughness_factor
            tree.links.new(roughness, scale.inputs[0])
            roughness = scale.outputs["Value"]
        tree.links.new(roughness, _socket(bsdf, "Roughness"))

    if maps.get("nor_gl"):
        normal_tex = box_texture(maps["nor_gl"], non_color=True)
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        # grazing views of a tiled floor amplify bump detail, so the strength is
        # configurable per surface (spec: materials.normal_strength)
        normal_map.inputs["Strength"].default_value = float(normal_strength)
        tree.links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
        tree.links.new(normal_map.outputs["Normal"], _socket(bsdf, "Normal"))

    _socket(bsdf, "Specular IOR Level", "Specular").default_value = 0.5
    material["ph_texture"] = texture_id
    material["ph_tile_m"] = tile
    return material


def build_scene_materials(cache_dir: str, spec, fallback_materials: dict) -> dict:
    """Resolve the spec's ``materials`` block into Blender materials.

    Returns ``{"floor": <material or None per room>, "wall": ..., "rug": ...}``;
    anything unspecified stays None so the P0 flat materials are kept.
    """
    block = getattr(spec, "materials", None) or {}
    normal = block.get("normal_strength", {}) or {}
    out = {"floor": {}, "wall": None, "ceiling": fallback_materials.get("ceiling"), "rug": None,
           "bedding": None, "source": {}}
    if block.get("wall"):
        out["wall"] = texture_material(
            cache_dir, assets_mod.split_asset_ref(block["wall"]), name="mat_wall_textured",
            tile_m=block.get("wall_tile_m"), normal_strength=float(normal.get("wall", 1.0)),
        )
        out["source"]["wall"] = block["wall"]
    if block.get("floor"):
        texture_id = assets_mod.split_asset_ref(block["floor"])
        out["floor_all"] = texture_material(
            cache_dir, texture_id, name="mat_floor_textured", tile_m=block.get("floor_tile_m"),
            normal_strength=float(normal.get("floor", 1.0)),
        )
        out["source"]["floor"] = block["floor"]
    if block.get("rug"):
        out["rug"] = texture_material(
            cache_dir, assets_mod.split_asset_ref(block["rug"]), name="mat_rug",
            tile_m=block.get("rug_tile_m"), normal_strength=float(normal.get("rug", 1.0)),
            tint=block.get("rug_tint"), tint_mix=float(block.get("rug_tint_mix", 0.55)),
        )
        out["source"]["rug"] = block["rug"]
    if block.get("bedding"):
        out["bedding"] = texture_material(cache_dir, assets_mod.split_asset_ref(block["bedding"]),
                                          name="mat_bedding", tile_m=block.get("bedding_tile_m"),
                                          normal_strength=float(normal.get("bedding", 1.0)))
        out["source"]["bedding"] = block["bedding"]
    return out


def apply_shell_materials(shell: dict, spec, materials: dict, fallback_materials: dict) -> dict:
    """Swap the shell's flat materials for textured ones where the spec asks."""
    applied = {}
    floor_mat = materials.get("floor_all")
    for obj in shell["floors"]:
        room_id = obj.name[len("floor_"):]
        if floor_mat is not None:
            obj.data.materials.clear()
            obj.data.materials.append(floor_mat)
            applied[room_id] = floor_mat.name
    if materials.get("wall") is not None:
        for obj in shell["walls"]:
            obj.data.materials.clear()
            obj.data.materials.append(materials["wall"])
    return applied
