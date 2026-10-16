"""P0 staging: Principled materials, Nishita sky world, sun lamp, sky dome.

The sky dome is a large sphere enclosing the whole house.  It exists so that
every camera ray ends on geometry: without it, rays that leave through a window
hit the world background, where Cycles has no surface position, and the depth
map would contain invalid (0) pixels.  The dome is visible to camera rays only
(``visible_shadow`` / ``visible_diffuse`` / ``visible_glossy`` are off), so the
interior is still lit by the world sky through the openings and the dome's own
emission is what a camera sees beyond the openings.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import bmesh
import bpy
from mathutils import Vector

from blender_agent import spec as spec_mod

ENVIRONMENT_COLLECTION = "environment"

WALL_COLOR = (0.9, 0.87, 0.82)
CEILING_COLOR = (1.0, 1.0, 1.0)
WALL_ROUGHNESS = 0.9
CEILING_ROUGHNESS = 0.9
FLOOR_ROUGHNESS = 0.6
DOME_RADIUS_MIN = 25.0
# Linear radiance values of the visible sky (the dome emits these; it does not
# light the room, the Nishita world does).  Kept well below the sun/sky radiance
# of the world so a window frame stays readable instead of blowing out.
GROUND_COLOR = (0.21, 0.22, 0.19)
HORIZON_COLOR = (0.55, 0.62, 0.70)
ZENITH_COLOR = (0.22, 0.38, 0.66)
SKY_DOME_STRENGTH = 0.85


def sun_direction(elevation_deg: float, azimuth_deg: float) -> Tuple[float, float, float]:
    """Unit vector pointing from the scene towards the sun.

    Azimuth is measured in the XY plane from +X towards +Y, elevation above the
    horizon - the same convention used for ``style.sun_elevation_deg`` /
    ``style.sun_azimuth_deg``.
    """
    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    return (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )


def make_principled(name: str, color, roughness: float) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    bsdf.inputs["Roughness"].default_value = float(roughness)
    bsdf.inputs["Metallic"].default_value = 0.0
    return material


def make_nishita_sky_node(node_tree: bpy.types.NodeTree, style: spec_mod.StyleSpec, vector=None):
    sky = node_tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(style.sun_elevation_deg)
    # Blender's Nishita sun_rotation is measured clockwise from +Y: a float-EXR
    # equirect probe of the world measured world_azimuth = 90 deg - sun_rotation,
    # so convert the spec's azimuth.
    sky.sun_rotation = math.radians(90.0 - style.sun_azimuth_deg)
    sky.altitude = 0.0
    if vector is not None:
        # the sky node's Vector socket is flagged disabled, so look it up by
        # index: name lookup raises even though the socket exists
        sky.inputs[0].default_value = (0.0, 0.0, 1.0)
        node_tree.links.new(vector, sky.inputs[0])
    return sky


def build_materials(spec: spec_mod.SceneSpec) -> Dict[str, bpy.types.Material]:
    materials = {
        "wall": make_principled("mat_wall", WALL_COLOR, WALL_ROUGHNESS),
        "ceiling": make_principled("mat_ceiling", CEILING_COLOR, CEILING_ROUGHNESS),
    }
    for room in spec.rooms:
        materials[f"floor_{room.id}"] = make_principled(
            f"mat_floor_{room.id}", room.floor_color, FLOOR_ROUGHNESS
        )
    return materials


def assign_shell_materials(shell: dict, spec: spec_mod.SceneSpec, materials: Dict[str, bpy.types.Material]) -> None:
    for room in spec.rooms:
        for obj in shell["floors"]:
            if obj.name == f"floor_{room.id}":
                obj.data.materials.append(materials[f"floor_{room.id}"])
        for obj in shell["ceilings"]:
            if obj.name == f"ceiling_{room.id}":
                obj.data.materials.append(materials["ceiling"])
    for wall in shell["walls"]:
        wall.data.materials.append(materials["wall"])
    # keep material slot bookkeeping explicit for the manifest/debugging
    for obj in shell["shell_objects"]:
        if not obj.data.materials:
            raise RuntimeError(f"shell object '{obj.name}' has no material assigned")


def setup_world(scene: bpy.types.Scene, style: spec_mod.StyleSpec) -> bpy.types.World:
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    node_tree = world.node_tree
    node_tree.nodes.clear()
    output = node_tree.nodes.new("ShaderNodeOutputWorld")
    background = node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = style.sky_strength
    sky = make_nishita_sky_node(node_tree, style)
    node_tree.links.new(sky.outputs["Color"], background.inputs["Color"])
    node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    return world


def setup_sun(scene: bpy.types.Scene, style: spec_mod.StyleSpec, energy: float = None) -> bpy.types.Object:
    """Sun lamp pointing from the same direction as the sky's sun."""
    direction = sun_direction(style.sun_elevation_deg, style.sun_azimuth_deg)
    lamp_data = bpy.data.lights.new("sun", type="SUN")
    lamp_data.energy = float(style.sun_energy if energy is None else energy)
    lamp_data.angle = math.radians(0.526)
    lamp = bpy.data.objects.new("sun", lamp_data)
    lamp.rotation_euler = Vector(direction).to_track_quat("Z", "Y").to_euler()
    scene.collection.objects.link(lamp)
    return lamp


def build_window_portals(spec: spec_mod.SceneSpec, energy: float, collection=None,
                         margin: float = 0.02) -> list:
    """Soft area lights just inside every window, aimed into the room.

    They stand in for the skylight that would otherwise have to be found by
    brute-force sampling through a small opening: a big soft source inside the
    reveal both lifts the interior and cuts render noise dramatically at a
    given sample count.
    """
    import mathutils

    lights = []
    for index, window in enumerate(spec.windows):
        p0 = (float(window.wall[0][0]), float(window.wall[0][1]))
        p1 = (float(window.wall[1][0]), float(window.wall[1][1]))
        length = math.dist(p0, p1)
        centre = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
        inward = spec_mod.inward_normal(window.room, centre, spec)
        offset = spec.wall_thickness / 2.0 + margin
        light_data = bpy.data.lights.new(f"window_portal_{index}", type="AREA")
        light_data.shape = "RECTANGLE"
        light_data.size = max(0.2, length - margin)
        light_data.size_y = max(0.2, window.height - margin)
        light_data.energy = float(energy)
        light_data.color = (0.96, 0.98, 1.0)
        light = bpy.data.objects.new(f"window_portal_{index}", light_data)
        light.location = (
            centre[0] + inward[0] * offset,
            centre[1] + inward[1] * offset,
            window.sill + window.height / 2.0,
        )
        # an area light emits along its local -Z; point that into the room
        forward = mathutils.Vector((inward[0], inward[1], 0.0)).normalized()
        light.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()
        # never visible to camera rays: it is lighting gear, not geometry, and
        # the depth validation raycasts against the exported geometry only
        light.visible_camera = False
        (collection or bpy.context.scene.collection).objects.link(light)
        lights.append(light)
    return lights


def build_lamp_lights(placements, energy: float, radius: float = 0.08, collection=None) -> list:
    """Point lights inside ceiling/side lamps (``placements`` = (x, y, z) list)."""
    lights = []
    for index, (x, y, z) in enumerate(placements):
        light_data = bpy.data.lights.new(f"lamp_light_{index}", type="POINT")
        light_data.energy = float(energy)
        light_data.shadow_soft_size = float(radius)
        light_data.color = (1.0, 0.94, 0.86)
        light = bpy.data.objects.new(f"lamp_light_{index}", light_data)
        light.location = (float(x), float(y), float(z))
        (collection or bpy.context.scene.collection).objects.link(light)
        lights.append(light)
    return lights


def build_sky_dome(
    spec: spec_mod.SceneSpec,
    style: spec_mod.StyleSpec,
    radius: float = None,
) -> bpy.types.Object:
    """Enclosing emissive sky sphere (see module docstring)."""
    xs = [p[0] for room in spec.rooms for p in room.polygon]
    ys = [p[1] for room in spec.rooms for p in room.polygon]
    center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, spec.wall_height / 2.0)
    diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    if radius is None:
        radius = max(DOME_RADIUS_MIN, 2.0 * diagonal)

    collection = bpy.data.collections.get(ENVIRONMENT_COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(ENVIRONMENT_COLLECTION)
        bpy.context.scene.collection.children.link(collection)

    mesh = bpy.data.meshes.new("sky_dome")
    bm = bmesh.new()
    try:
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=radius)
    except TypeError:  # Blender < 3.0 used `diameter`
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, diameter=radius)
    bm.to_mesh(mesh)
    bm.free()
    for vertex in mesh.vertices:
        vertex.co += Vector(center)

    dome = bpy.data.objects.new("sky_dome", mesh)
    collection.objects.link(dome)
    # Visible to camera rays (depth + visual sky) but not an occluder for light.
    dome.visible_camera = True
    dome.visible_diffuse = False
    dome.visible_glossy = False
    dome.visible_transmission = False
    dome.visible_volume_scatter = False
    dome.visible_shadow = False

    material = bpy.data.materials.new("mat_sky_dome")
    material.use_nodes = True
    node_tree = material.node_tree
    node_tree.nodes.clear()
    output = node_tree.nodes.new("ShaderNodeOutputMaterial")
    emission = node_tree.nodes.new("ShaderNodeEmission")

    # Hand-built sky gradient. A Nishita Sky Texture node cannot be used here:
    # in a material its Vector socket is disabled and Cycles ignores links into
    # it (measured: linked and unlinked renders are bit-identical), so the
    # gradient is composed from the view direction instead.
    #
    # Measured convention (float-EXR probe of the dome): Cycles' Geometry
    # "Incoming" at a camera-ray hit is the direction from the surface *back
    # towards the camera*, i.e. -view_dir.  Everything below therefore works in
    # terms of Incoming only, with float Math nodes (a VectorMath SCALE with a
    # scalar assignment silently misbehaves on vector sockets).
    geometry = node_tree.nodes.new("ShaderNodeNewGeometry")
    sun_dir = sun_direction(style.sun_elevation_deg, style.sun_azimuth_deg)

    sun_dot = node_tree.nodes.new("ShaderNodeVectorMath")
    sun_dot.operation = "DOT_PRODUCT"
    sun_dot.inputs[1].default_value = (-sun_dir[0], -sun_dir[1], -sun_dir[2])
    node_tree.links.new(geometry.outputs["Incoming"], sun_dot.inputs[0])

    split = node_tree.nodes.new("ShaderNodeSeparateXYZ")
    node_tree.links.new(geometry.outputs["Incoming"], split.inputs[0])
    flip_height = node_tree.nodes.new("ShaderNodeMath")
    flip_height.operation = "MULTIPLY"
    flip_height.inputs[1].default_value = -1.0
    node_tree.links.new(split.outputs["Z"], flip_height.inputs[0])

    height = node_tree.nodes.new("ShaderNodeMapRange")
    height.inputs["From Min"].default_value = -0.02
    height.inputs["From Max"].default_value = 0.7
    height.clamp = True
    node_tree.links.new(flip_height.outputs["Value"], height.inputs["Value"])

    ramp = node_tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "EASE"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (GROUND_COLOR[0], GROUND_COLOR[1], GROUND_COLOR[2], 1.0)
    horizon = ramp.color_ramp.elements.new(0.11)
    horizon.color = (HORIZON_COLOR[0], HORIZON_COLOR[1], HORIZON_COLOR[2], 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (ZENITH_COLOR[0], ZENITH_COLOR[1], ZENITH_COLOR[2], 1.0)
    middle = ramp.color_ramp.elements.new(0.45)
    middle.color = (
        (HORIZON_COLOR[0] + ZENITH_COLOR[0]) / 2.0,
        (HORIZON_COLOR[1] + ZENITH_COLOR[1]) / 2.0,
        (HORIZON_COLOR[2] + ZENITH_COLOR[2]) / 2.0,
        1.0,
    )
    node_tree.links.new(height.outputs["Result"], ramp.inputs["Fac"])

    glow_power = node_tree.nodes.new("ShaderNodeMath")
    glow_power.operation = "POWER"
    glow_power.inputs[1].default_value = 320.0
    # clamp first: an even exponent would make the anti-solar hemisphere glow too
    glow_clamp = node_tree.nodes.new("ShaderNodeMath")
    glow_clamp.operation = "MAXIMUM"
    glow_clamp.inputs[1].default_value = 0.0
    node_tree.links.new(sun_dot.outputs["Value"], glow_clamp.inputs[0])
    node_tree.links.new(glow_clamp.outputs["Value"], glow_power.inputs[0])
    glow_amount = node_tree.nodes.new("ShaderNodeMath")
    glow_amount.operation = "MULTIPLY"
    glow_amount.inputs[1].default_value = 1.4
    node_tree.links.new(glow_power.outputs["Value"], glow_amount.inputs[0])

    add_glow = node_tree.nodes.new("ShaderNodeMixRGB")
    add_glow.blend_type = "ADD"
    add_glow.inputs["Fac"].default_value = 1.0
    add_glow.inputs["Color2"].default_value = (1.0, 0.95, 0.85, 1.0)
    node_tree.links.new(ramp.outputs["Color"], add_glow.inputs["Color1"])
    node_tree.links.new(glow_amount.outputs["Value"], add_glow.inputs["Fac"])

    node_tree.links.new(add_glow.outputs["Color"], emission.inputs["Color"])
    emission.inputs["Strength"].default_value = style.sky_strength * SKY_DOME_STRENGTH
    node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    mesh.materials.append(material)

    dome["p0_dome_radius"] = float(radius)
    dome["p0_dome_center"] = [float(v) for v in center]
    return dome


def setup_stage(scene: bpy.types.Scene, spec: spec_mod.SceneSpec, shell: dict,
                materials: dict = None, apply_materials: bool = True) -> dict:
    materials = materials or build_materials(spec)
    if apply_materials:
        assign_shell_materials(shell, spec, materials)
    world = setup_world(scene, spec.style)
    sun = setup_sun(scene, spec.style)
    dome = build_sky_dome(spec, spec.style)
    return {
        "materials": materials,
        "world": world,
        "sun": sun,
        "dome": dome,
        "dome_radius": float(dome["p0_dome_radius"]),
        "dome_center": list(dome["p0_dome_center"]),
        "environment_objects": [dome],
    }
